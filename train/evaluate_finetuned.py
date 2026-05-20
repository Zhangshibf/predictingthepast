"""Evaluate model A (fine-tuned Aeneas) on the test set.

Inputs
------
  aeneas_test_windows.json     # {doc_name, text, partition_spans}
  damage_spans_aeneas.json     # {doc_name: [{start, end, tag, length_class}, ...]}
  <checkpoint.pkl>             # in the released "flat" schema:
                                 {'params', 'model_config', 'region_map'}

Two evaluation settings, controlled by --setting:
  known    (length-known):   each damage [s, e) -> '?' * (e - s)
  unknown  (length-unknown): each damage [s, e) -> '#'  (one token)

Per-damage outputs are collected as PredictionResult records. We then report:
  - Mean CER and top-N hit rate across ALL damage spans.
  - Mean CER and top-N hit rate PER tag (e.g. 't', 'p').

Scoring rules (decided with the user beforehand):
  - CER = Levenshtein(pred, gt) / max(len(pred), len(gt))    -- handles unequal
    lengths in setting 'unknown'. CER for empty pred and empty gt is 0.0.
  - Top-N hit rate = fraction of damages where the ground-truth string appears
    exactly among the top-N predictions (strict string match).

Aggregation: each damage span is one data point (option (a)).

Usage
-----
  python evaluate_aeneas.py \
      --checkpoint /path/to/checkpoint.pkl \
      --windows aeneas_test_windows.json \
      --damages damage_spans_aeneas.json \
      --setting known \
      --topn 5 \
      --out results_known.json

Or run both settings in one go:
  python evaluate_aeneas.py --checkpoint ... --setting both ...
  (writes results_known.json and results_unknown.json side by side)
"""

import argparse
import json
import os
import pickle
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

import jax

from predictingthepast.eval import inference
from predictingthepast.models.model import Model
from predictingthepast.util import alphabet as util_alphabet


# --------------------------------------------------------------------------
# Result dataclass (matches the schema the user specified)
# --------------------------------------------------------------------------


@dataclass
class PredictionResult:
    """Container for prediction results."""
    original_text: str
    masked_text: str
    predicted_sequence: str
    predicted_sequences: List[Tuple[str, float]]  # (sequence, score)
    masked_span: Tuple[int, int]
    ground_truth: Optional[str] = None
    # Extra fields added at the user's request, kept after the dataclass
    # defaults so existing callers using positional args still work.
    tag: Optional[str] = None
    length_class: Optional[str] = None
    doc_name: Optional[str] = None
    setting: Optional[str] = None     # 'known' or 'unknown'


# --------------------------------------------------------------------------
# Levenshtein-based CER
# --------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    """Plain DP edit distance. We don't import a library to keep deps light;
    these strings are short (typically <= 30 chars). O(len(a)*len(b)) is fine."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Two-row DP.
    prev = list(range(len(b) + 1))
    cur = [0] * (len(b) + 1)
    for i, ca in enumerate(a, 1):
        cur[0] = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1,        # deletion
                         cur[j - 1] + 1,     # insertion
                         prev[j - 1] + cost) # substitution
        prev, cur = cur, prev
    return prev[len(b)]


def cer(pred: str, gt: str) -> float:
    """Normalized edit distance: edit / max(len). 0.0 if both empty."""
    if not pred and not gt:
        return 0.0
    return _levenshtein(pred, gt) / max(len(pred), len(gt))


# --------------------------------------------------------------------------
# Checkpoint loading
#
# Two schemas are supported:
#
#  1. Flat (inference) schema -- what the released aeneas_117149994_2.pkl is.
#     A dict: {'params': <pytree>, 'model_config': <dict>, 'region_map': ...}.
#     `params` is host-side (un-replicated). `model_config` directly
#     constructs the model.
#
#  2. Jaxline snapshot schema -- what our DiskCheckpointer writes.
#     A list with one SnapshotNT(id, pickle_nest), where pickle_nest is a
#     ConfigDict containing experiment_module with snapshotted CHECKPOINT_ATTRS.
#     `params` is REPLICATED across local devices (leading device axis added
#     by pmap during training), so we de-replicate. `model_config` is not in
#     the snapshot -- we read it from the config file passed alongside.
# --------------------------------------------------------------------------

def _is_jaxline_snapshot(obj):
    """Heuristic: a list whose first element looks like SnapshotNT."""
    if not isinstance(obj, list) or not obj:
        return False
    first = obj[0]
    return (hasattr(first, '_fields')
            and 'id' in first._fields
            and 'pickle_nest' in first._fields)


def _extract_params_from_snapshot(snapshot_list):
    """Pull host-side, un-replicated params out of a jaxline snapshot list."""
    snap = snapshot_list[-1]  # one snapshot in our files; take latest anyway.
    nest = snap.pickle_nest
    if hasattr(nest, 'to_dict'):
        nest = nest.to_dict()
    exp_state = nest.get('experiment_module', {})
    if hasattr(exp_state, 'to_dict'):
        exp_state = exp_state.to_dict()
    params = exp_state.get('_params') or exp_state.get('params')
    if params is None:
        raise KeyError(
            "Snapshot has no '_params' / 'params' in experiment_module. "
            f"Keys present: {list(exp_state.keys())}")
    # Params are replicated across local devices (leading axis); take device 0.
    params = jax.tree_util.tree_map(
        lambda x: x[0] if hasattr(x, 'ndim') and x.ndim > 0 else x, params)
    return params


def load_checkpoint(path, model_config_override=None, language='latin'):
    """Load params + build forward fn from either a flat or jaxline checkpoint.

    Args:
      path: pickle file path.
      model_config_override: dict for Model(**config). REQUIRED when the
        checkpoint is a jaxline snapshot (snapshots do not carry the model
        config); IGNORED when the checkpoint is flat.
    """
    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)

    if _is_jaxline_snapshot(checkpoint):
        if model_config_override is None:
            raise ValueError(
                f"{path} is a jaxline snapshot; pass --config so model_config "
                f"can be read from it.")
        params = _extract_params_from_snapshot(checkpoint)
        model_config = dict(model_config_override)
    elif isinstance(checkpoint, dict) and 'params' in checkpoint \
            and 'model_config' in checkpoint:
        params = checkpoint['params']
        model_config = checkpoint['model_config']
    else:
        kind = type(checkpoint).__name__
        keys = (list(checkpoint.keys())
                if isinstance(checkpoint, dict) else 'N/A')
        raise ValueError(
            f"{path}: unrecognised checkpoint schema (type={kind}, keys={keys}).")

    params = jax.device_put(params)
    model = Model(**model_config)
    forward = model.apply
    if language == 'latin':
        alphabet = util_alphabet.LatinAlphabet()
    elif language == 'greek':
        alphabet = util_alphabet.GreekAlphabet()
    else:
        raise ValueError(f'Unknown language: {language}')
    vocab_char_size = model_config['vocab_char_size']
    return params, forward, alphabet, vocab_char_size


# --------------------------------------------------------------------------
# Building the damaged window input
# --------------------------------------------------------------------------

def build_damaged_text(window_text, damages, setting):
    """Apply all damages to window_text.

    setting='known':   damage [s, e) -> '?' * (e - s)
    setting='unknown': damage [s, e) -> '#' (single char)

    Returns (damaged_text, damage_records) where damage_records lists each
    damage in the SAME ORDER they appear in the damaged text, augmented with
    the position of its mask token(s) in damaged_text:

      [{
        'orig': original substring,
        'tag': ...,
        'length_class': ...,
        'damaged_start': position of first mask char in damaged_text,
        'damaged_end': one past last mask char in damaged_text,
        'orig_start': original start in window_text,
        'orig_end': original end in window_text,
      }, ...]

    We sort damages by start position before applying so positions in the
    output can be computed in a single left-to-right sweep.
    """
    assert setting in ('known', 'unknown')

    # Sort by source position; damages don't overlap by construction
    # (create_aeneas_test_set guarantees this).
    sorted_dmg = sorted(damages, key=lambda d: d['start'])

    out = []
    records = []
    cursor = 0
    for d in sorted_dmg:
        s, e = d['start'], d['end']
        if s < cursor:
            # Should not happen given non-overlap, but skip defensively.
            continue
        out.append(window_text[cursor:s])
        damaged_start = sum(len(p) for p in out)
        if setting == 'known':
            mask = inference.ALPHABET_MISSING_RESTORE * (e - s)  # '?' * n
        else:
            mask = inference.ALPHABET_MISSING_UNK_RESTORE        # '#'
        out.append(mask)
        damaged_end = damaged_start + len(mask)
        records.append({
            'orig': window_text[s:e],
            'tag': d.get('tag'),
            'length_class': d.get('length_class'),
            'damaged_start': damaged_start,
            'damaged_end': damaged_end,
            'orig_start': s,
            'orig_end': e,
        })
        cursor = e
    out.append(window_text[cursor:])
    return ''.join(out), records


# --------------------------------------------------------------------------
# Extracting per-damage predictions from a RestorationResults
# --------------------------------------------------------------------------

def _split_restored_into_runs(restored_indices):
    """Group a sorted list of monotonically-increasing ints into maximal
    runs of consecutive integers. Returns [(start, end_inclusive), ...]."""
    if not restored_indices:
        return []
    runs = []
    s = p = restored_indices[0]
    for i in restored_indices[1:]:
        if i == p + 1:
            p = i
        else:
            runs.append((s, p))
            s = p = i
    runs.append((s, p))
    return runs


def extract_predictions_for_damages(restoration, damage_records, setting):
    """Pull one (top_pred_string, per_candidate_list) per damage from a
    RestorationResults. Order of returned predictions matches damage_records.

    setting='known': Restoration.text has the same length as damaged_text;
        per-damage prediction is the slice [damaged_start, damaged_end).
    setting='unknown': Each '#' may expand to multiple chars in the output;
        we identify per-damage runs by splitting Restoration.restored at
        gaps, in order.
    """
    assert setting in ('known', 'unknown')

    # Per-candidate per-damage strings.
    # candidates[i] = list of (pred_string_for_damage_i, score), one entry
    #                 per beam candidate.
    candidates = [[] for _ in damage_records]

    for restoration_candidate in restoration.predictions:
        pred_text = restoration_candidate.text
        score = restoration_candidate.score

        if setting == 'known':
            # Slice each damage's range out of pred_text directly.
            for i, rec in enumerate(damage_records):
                ds, de = rec['damaged_start'], rec['damaged_end']
                # Safety: index might go out of bounds if pred_text was
                # truncated for some reason; clip.
                if de <= len(pred_text):
                    candidates[i].append((pred_text[ds:de], score))
                else:
                    candidates[i].append((pred_text[ds:], score))
        else:
            # Group restored indices into runs; one run per source '#'.
            runs = _split_restored_into_runs(restoration_candidate.restored)
            if len(runs) != len(damage_records):
                # This shouldn't happen if the input had exactly N '#'s --
                # the model is asked to restore exactly N runs. Warn loudly
                # rather than silently misalign predictions to damages.
                print(f"  WARNING: candidate has {len(runs)} restored runs "
                      f"but expected {len(damage_records)} damages. "
                      f"Skipping this candidate.", file=sys.stderr)
                continue
            for i, (rs, re_inc) in enumerate(runs):
                candidates[i].append((pred_text[rs:re_inc + 1], score))

    # Per-damage top prediction = the first non-empty candidate, which is
    # the highest-scoring one (RestorationResults.predictions is sorted).
    top_preds = []
    for cand_list in candidates:
        top_preds.append(cand_list[0][0] if cand_list else '')

    return top_preds, candidates


# --------------------------------------------------------------------------
# Main evaluation loop for one setting
# --------------------------------------------------------------------------

def _debug_print_window(doc_name, setting, text, damaged_text, dmg_records,
                        results_for_this_window, topn):
    """Per-window verbose dump (smoke-test mode)."""
    print(f"\n  ---- {doc_name}  [setting={setting}] ----")
    print(f"  window_len={len(text)}  damaged_len={len(damaged_text)}  "
          f"n_damages={len(dmg_records)}")

    # Show each damage region in context, so we can eyeball that the
    # mask tokens landed where we expected. Print a ~30-char window
    # of damaged text around each damage, with the mask portion bracketed.
    pad = 25
    for i, rec in enumerate(dmg_records):
        ds, de = rec['damaged_start'], rec['damaged_end']
        left = max(0, ds - pad)
        right = min(len(damaged_text), de + pad)
        before = damaged_text[left:ds]
        mask = damaged_text[ds:de]
        after = damaged_text[de:right]
        print(f"    dmg[{i}] tag={rec['tag']!r} class={rec['length_class']!r} "
              f"orig=[{rec['orig_start']}:{rec['orig_end']}) len="
              f"{rec['orig_end'] - rec['orig_start']}")
        print(f"      gt:     {rec['orig']!r}")
        print(f"      context: ...{before}[{mask}]{after}...")

    # Per-damage prediction quality.
    for i, r in enumerate(results_for_this_window):
        gt = r.ground_truth or ''
        pred = r.predicted_sequence
        this_cer = cer(pred, gt)
        cand_strs = [c[0] for c in r.predicted_sequences[:topn]]
        hit = gt in cand_strs
        # Top-3 with scores, more is too noisy for a smoke test.
        top3 = ', '.join(f'{c[0]!r}@{c[1]:.3f}'
                         for c in r.predicted_sequences[:3])
        print(f"    pred[{i}] top1={pred!r}  CER={this_cer:.3f}  "
              f"top{topn}_hit={'YES' if hit else 'no '}")
        print(f"      top3: {top3}")


def evaluate_setting(windows, damages_by_doc, setting, forward, params,
                     alphabet, vocab_char_size, topn=5, beam_width=200,
                     unk_max_len=20, max_windows=None, verbose=True,
                     debug=False):
    """Returns (results: list[PredictionResult], aggregate: dict).

    When debug=True, prints detailed per-window information: damage
    positions, masked-text context, top predictions, per-damage CER, and
    top-N hit. Intended for smoke-testing with --max-windows.
    """
    results = []
    n_windows_used = 0
    n_windows_skipped = 0
    n_damages_total = 0
    n_failures = 0
    t0 = time.time()

    for w_idx, win in enumerate(windows):
        if max_windows is not None and n_windows_used >= max_windows:
            break

        doc_name = win['doc_name']
        text = win['text']
        damages = damages_by_doc.get(doc_name, [])
        if not damages:
            continue

        damaged_text, dmg_records = build_damaged_text(text, damages, setting)

        # Skip if too short or too long for the model.
        if len(damaged_text) < inference.MIN_TEXT_LEN:
            n_windows_skipped += 1
            if debug:
                print(f"  [skip] {doc_name}: damaged_text too short "
                      f"({len(damaged_text)} < {inference.MIN_TEXT_LEN})")
            continue
        if len(damaged_text) >= inference.TEXT_LEN:
            n_windows_skipped += 1
            if verbose or debug:
                print(f"  [skip] {doc_name}: damaged_text too long "
                      f"({len(damaged_text)} >= {inference.TEXT_LEN})")
            continue

        try:
            restoration = inference.restore(
                damaged_text,
                forward=forward,
                params=params,
                alphabet=alphabet,
                vocab_char_size=vocab_char_size,
                beam_width=beam_width,
                unk_restoration_max_len=unk_max_len,
            )
        except Exception as e:
            n_failures += 1
            if verbose or debug:
                print(f"  restore() failed on {doc_name}: {e}",
                      file=sys.stderr)
            continue

        top_preds, candidates = extract_predictions_for_damages(
            restoration, dmg_records, setting)

        results_for_this_window = []
        for rec, top_pred, cand_list in zip(dmg_records, top_preds, candidates):
            top_cands = cand_list[:topn]
            r = PredictionResult(
                original_text=text,
                masked_text=damaged_text,
                predicted_sequence=top_pred,
                predicted_sequences=top_cands,
                masked_span=(rec['orig_start'], rec['orig_end']),
                ground_truth=rec['orig'],
                tag=rec['tag'],
                length_class=rec['length_class'],
                doc_name=doc_name,
                setting=setting,
            )
            results.append(r)
            results_for_this_window.append(r)
            n_damages_total += 1

        if debug:
            _debug_print_window(doc_name, setting, text, damaged_text,
                                dmg_records, results_for_this_window, topn)

        n_windows_used += 1
        if verbose and not debug and n_windows_used % 50 == 0:
            elapsed = time.time() - t0
            rate = n_windows_used / max(elapsed, 1e-6)
            print(f"  [{setting}] {n_windows_used} windows / "
                  f"{n_damages_total} damages in {elapsed:.1f}s "
                  f"({rate:.2f} win/s)")

    aggregate = aggregate_metrics(results, topn=topn)
    aggregate['n_windows_used'] = n_windows_used
    aggregate['n_windows_skipped'] = n_windows_skipped
    aggregate['n_failures'] = n_failures
    aggregate['n_damages_total'] = n_damages_total
    aggregate['elapsed_s'] = time.time() - t0
    return results, aggregate


# --------------------------------------------------------------------------
# Aggregation: overall + per-tag
# --------------------------------------------------------------------------

def aggregate_metrics(results, topn=5):
    def _agg(items):
        if not items:
            return {'n': 0, 'cer': None, f'top{topn}_hit': None}
        cers = []
        hits = 0
        for r in items:
            pred = r.predicted_sequence
            gt = r.ground_truth or ''
            cers.append(cer(pred, gt))
            cand_strs = [c[0] for c in (r.predicted_sequences or [])][:topn]
            if gt in cand_strs:
                hits += 1
        return {
            'n': len(items),
            'cer': statistics.fmean(cers),
            f'top{topn}_hit': hits / len(items),
        }

    overall = _agg(results)

    by_tag = {}
    tagged = defaultdict(list)
    for r in results:
        tagged[r.tag or '<none>'].append(r)
    for tag, items in tagged.items():
        by_tag[tag] = _agg(items)

    return {'overall': overall, 'by_tag': by_tag}


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def _print_report(setting, agg, topn):
    print(f"\n=== Setting: {setting} ===")
    print(f"windows used:      {agg['n_windows_used']}")
    print(f"windows skipped:   {agg['n_windows_skipped']}")
    print(f"restore failures:  {agg['n_failures']}")
    print(f"damages scored:    {agg['n_damages_total']}")
    print(f"elapsed:           {agg['elapsed_s']:.1f}s")
    ov = agg['overall']
    if ov['n']:
        print(f"\nOverall: CER={ov['cer']:.4f}  "
              f"top{topn}_hit={ov[f'top{topn}_hit']:.4f}  (n={ov['n']})")
    print(f"\nBy tag:")
    for tag, m in sorted(agg['by_tag'].items()):
        if m['n']:
            print(f"  tag={tag!r:>6}  n={m['n']:4d}  "
                  f"CER={m['cer']:.4f}  top{topn}_hit={m[f'top{topn}_hit']:.4f}")


def _dump_results(out_path, results, aggregate):
    payload = {
        'aggregate': aggregate,
        'results': [asdict(r) for r in results],
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True,
                        help='Path to a checkpoint pickle. Accepts both the '
                             'released flat schema (dict with "params" + '
                             '"model_config") and jaxline DiskCheckpointer '
                             'snapshots (list of SnapshotNT). For snapshots, '
                             '--config is required to read the model config.')
    parser.add_argument('--config',
                        help='Path to config_paleo_eval.py. REQUIRED if '
                             '--checkpoint is a jaxline snapshot. The '
                             'model_config is taken from '
                             'config.experiment_kwargs.config.model.')
    parser.add_argument('--windows', required=True,
                        help='aeneas_test_windows.json')
    parser.add_argument('--damages', required=True,
                        help='damage_spans_aeneas.json')
    parser.add_argument('--setting', choices=['known', 'unknown', 'both'],
                        default='both')
    parser.add_argument('--topn', type=int, default=5)
    parser.add_argument('--beam-width', type=int,
                        default=inference.RESTORATION_BEAM_WIDTH)
    parser.add_argument('--unk-max-len', type=int,
                        default=inference.UNK_RESTORATION_MAX_LEN,
                        help=f'Max chars per # in setting=unknown. '
                             f'Default {inference.UNK_RESTORATION_MAX_LEN}.')
    parser.add_argument('--out-prefix', default='results',
                        help="Output files are written to "
                             "<prefix>_<setting>.json. Default: 'results'.")
    parser.add_argument('--max-windows', type=int, default=None,
                        help='Stop after this many windows (for debugging).')
    parser.add_argument('--language', default='latin', choices=['latin', 'greek'])
    args = parser.parse_args()

    # If a config file is provided, read the model architecture out of it.
    # This is REQUIRED when the checkpoint is a jaxline snapshot
    # (DiskCheckpointer output); IGNORED when it's a flat inference pickle.
    model_config_override = None
    if args.config:
        print(f"Loading config: {args.config}")
        import importlib.util
        spec = importlib.util.spec_from_file_location("eval_config", args.config)
        cfg_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg_mod)
        cfg = cfg_mod.get_config()
        model_config_override = dict(cfg.experiment_kwargs.config.model)
        print(f"  read model_config "
              f"(emb_dim={model_config_override.get('emb_dim')}, "
              f"num_layers={model_config_override.get('num_layers')})")

    print(f"Loading checkpoint: {args.checkpoint}")
    params, forward, alphabet, vocab_char_size = load_checkpoint(
        args.checkpoint,
        model_config_override=model_config_override,
        language=args.language,
    )

    print(f"Loading windows: {args.windows}")
    with open(args.windows, 'r', encoding='utf-8') as f:
        windows = json.load(f)
    print(f"  {len(windows)} windows.")

    print(f"Loading damages: {args.damages}")
    with open(args.damages, 'r', encoding='utf-8') as f:
        damages_by_doc = json.load(f)
    print(f"  {len(damages_by_doc)} doc_names; "
          f"{sum(len(v) for v in damages_by_doc.values())} damages total.")

    settings = (['known', 'unknown'] if args.setting == 'both'
                else [args.setting])

    # When --max-windows is set, this is a smoke test: turn on per-window
    # debug output so the user can verify alignment and predictions by eye
    # before launching the full run.
    debug = args.max_windows is not None
    if debug:
        print(f"\n[DEBUG MODE] --max-windows={args.max_windows} -> "
              f"printing detailed per-window output.")

    for setting in settings:
        print(f"\n----- Evaluating setting '{setting}' -----")
        results, aggregate = evaluate_setting(
            windows, damages_by_doc, setting,
            forward, params, alphabet, vocab_char_size,
            topn=args.topn, beam_width=args.beam_width,
            unk_max_len=args.unk_max_len,
            max_windows=args.max_windows,
            debug=debug,
        )
        _print_report(setting, aggregate, args.topn)
        out_path = f"{args.out_prefix}_{setting}.json"
        _dump_results(out_path, results, aggregate)
        print(f"\nWrote {out_path}  ({len(results)} damages, "
              f"{os.path.getsize(out_path)} bytes)")


if __name__ == '__main__':
    main()