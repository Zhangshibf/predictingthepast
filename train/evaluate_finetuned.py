"""Evaluate a fine-tuned Aeneas checkpoint on the damaged test set.

This script is a direct adaptation of inference_example.py:
  * Uses the same load_checkpoint logic (flat-schema pickle only).
  * Uses the same `forward = model.apply` pattern.
  * Calls inference.restore() the same way.

The differences from inference_example:
  * loops over many windows in aeneas_test_windows.json instead of a
    single --input string;
  * for each window, applies a list of damages from damage_spans_aeneas.json
    (replacing each span with `?` * length in setting='known', or a single
    `#` in setting='unknown') before calling restore();
  * extracts per-damage predictions from the returned RestorationResults
    and stores them in PredictionResult records;
  * aggregates: overall mean CER and top-N hit rate, and the same broken
    down per tag.

Required input: a FLAT checkpoint. If yours is a jaxline DiskCheckpointer
snapshot (i.e. a list of SnapshotNT), run convert_snapshot_to_flat.py first.

Scoring rules:
  CER = Levenshtein(pred, gt) / max(len(pred), len(gt))
        -- handles unequal lengths in setting='unknown'.
  Top-N hit = ground-truth string appears EXACTLY among the top-N candidates.

Aggregation: each damage span is one data point. Reports overall + per-tag.
"""

import argparse
import csv
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
# Result dataclass (per damage span)
# --------------------------------------------------------------------------


@dataclass
class PredictionResult:
    """One per damage span. Mirrors the user's specified schema, plus
    a few metadata fields appended."""
    original_text: str
    masked_text: str
    predicted_sequence: str
    predicted_sequences: List[Tuple[str, float]]
    masked_span: Tuple[int, int]
    ground_truth: Optional[str] = None
    tag: Optional[str] = None
    length_class: Optional[str] = None
    doc_name: Optional[str] = None
    setting: Optional[str] = None


# --------------------------------------------------------------------------
# Checkpoint loading (verbatim from inference_example.load_checkpoint)
# --------------------------------------------------------------------------


def load_checkpoint(path, language):
    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)
    if not isinstance(checkpoint, dict) or 'params' not in checkpoint \
            or 'model_config' not in checkpoint:
        kind = type(checkpoint).__name__
        keys = list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'N/A'
        raise ValueError(
            f"{path}: not a flat (inference) checkpoint (type={kind}, keys="
            f"{keys}). If this is a jaxline snapshot, run "
            f"convert_snapshot_to_flat.py first.")
    params = jax.device_put(checkpoint['params'])
    model = Model(**checkpoint['model_config'])
    forward = model.apply
    region_map = checkpoint['region_map']
    if language == 'latin':
        alphabet = util_alphabet.LatinAlphabet()
    elif language == 'greek':
        alphabet = util_alphabet.GreekAlphabet()
    else:
        raise ValueError(f'Unknown language: {language}')
    return checkpoint['model_config'], region_map, alphabet, params, forward


# --------------------------------------------------------------------------
# Levenshtein-based CER
# --------------------------------------------------------------------------


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    cur = [0] * (len(b) + 1)
    for i, ca in enumerate(a, 1):
        cur[0] = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return prev[len(b)]


def cer(pred: str, gt: str) -> float:
    if not pred and not gt:
        return 0.0
    return _levenshtein(pred, gt) / max(len(pred), len(gt))


# --------------------------------------------------------------------------
# Damage application
# --------------------------------------------------------------------------


def build_damaged_text(window_text, damages, setting):
    """Apply all damages to window_text.

      setting='known':   damage [s, e) -> '?' * (e - s)
      setting='unknown': damage [s, e) -> '#'

    Returns (damaged_text, records). records[i] contains positions of the
    mask token(s) in damaged_text along with the original substring and
    metadata. Records are in damaged_text order (i.e. sorted by start).
    """
    assert setting in ('known', 'unknown')
    sorted_dmg = sorted(damages, key=lambda d: d['start'])

    parts = []
    records = []
    cursor = 0
    for d in sorted_dmg:
        s, e = d['start'], d['end']
        if s < cursor:
            continue  # overlapping; skip defensively
        parts.append(window_text[cursor:s])
        damaged_start = sum(len(p) for p in parts)
        if setting == 'known':
            mask = inference.ALPHABET_MISSING_RESTORE * (e - s)
        else:
            mask = inference.ALPHABET_MISSING_UNK_RESTORE
        parts.append(mask)
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
    parts.append(window_text[cursor:])
    return ''.join(parts), records


# --------------------------------------------------------------------------
# Per-damage prediction extraction
# --------------------------------------------------------------------------


def _split_restored_into_runs(restored_indices):
    """Group monotonically-increasing ints into maximal consecutive runs."""
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
    """For each damage, pull out the predicted substring and the list of
    (substring, score) candidates from the top beams.

    setting='known':   Restoration.text has the same length as the damaged
                       input, so per-damage prediction is a fixed slice
                       [damaged_start, damaged_end).
    setting='unknown': Each '#' expands to a variable-length run in the
                       output. Group Restoration.restored into consecutive
                       runs (one per '#') and slice Restoration.text by
                       run boundaries.
    """
    assert setting in ('known', 'unknown')

    candidates = [[] for _ in damage_records]
    for cand in restoration.predictions:
        pred_text = cand.text
        score = cand.score
        if setting == 'known':
            for i, rec in enumerate(damage_records):
                ds, de = rec['damaged_start'], rec['damaged_end']
                end = min(de, len(pred_text))
                candidates[i].append((pred_text[ds:end], score))
        else:
            runs = _split_restored_into_runs(cand.restored)
            if len(runs) != len(damage_records):
                print(f"  WARNING: candidate has {len(runs)} restored runs "
                      f"but expected {len(damage_records)} damages; "
                      f"skipping this candidate.", file=sys.stderr)
                continue
            for i, (rs, re_inc) in enumerate(runs):
                candidates[i].append((pred_text[rs:re_inc + 1], score))

    top_preds = [c[0][0] if c else '' for c in candidates]
    return top_preds, candidates


# --------------------------------------------------------------------------
# Debug printout (smoke-test mode)
# --------------------------------------------------------------------------


def _debug_print_window(doc_name, setting, text, damaged_text, dmg_records,
                        results_for_this_window, topn):
    print(f"\n  ---- {doc_name}  [setting={setting}] ----")
    print(f"  window_len={len(text)}  damaged_len={len(damaged_text)}  "
          f"n_damages={len(dmg_records)}")
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
        print(f"      gt:      {rec['orig']!r}")
        print(f"      context: ...{before}[{mask}]{after}...")

    for i, r in enumerate(results_for_this_window):
        gt = r.ground_truth or ''
        pred = r.predicted_sequence
        this_cer = cer(pred, gt)
        cand_strs = [c[0] for c in (r.predicted_sequences or [])][:topn]
        hit = gt in cand_strs
        top3 = ', '.join(f'{c[0]!r}@{c[1]:.3f}'
                         for c in r.predicted_sequences[:3])
        print(f"    pred[{i}] top1={pred!r}  CER={this_cer:.3f}  "
              f"top{topn}_hit={'YES' if hit else 'no '}")
        print(f"      top3: {top3}")


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def aggregate_metrics(results, topn=5):
    def _agg(items):
        if not items:
            return {'n': 0, 'cer': None, f'top{topn}_hit': None}
        cers, hits = [], 0
        for r in items:
            gt = r.ground_truth or ''
            cers.append(cer(r.predicted_sequence, gt))
            cand_strs = [c[0] for c in (r.predicted_sequences or [])][:topn]
            if gt in cand_strs:
                hits += 1
        return {'n': len(items),
                'cer': statistics.fmean(cers),
                f'top{topn}_hit': hits / len(items)}

    overall = _agg(results)
    by_tag = {}
    tagged = defaultdict(list)
    for r in results:
        tagged[r.tag or '<none>'].append(r)
    for tag, items in tagged.items():
        by_tag[tag] = _agg(items)
    return {'overall': overall, 'by_tag': by_tag}


# --------------------------------------------------------------------------
# Main per-setting loop
# --------------------------------------------------------------------------


def evaluate_setting(windows, damages_by_doc, setting, forward, params,
                     alphabet, vocab_char_size, topn, beam_width,
                     unk_max_len, max_windows, debug, temperature=1.0):
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

        if len(damaged_text) < inference.MIN_TEXT_LEN:
            n_windows_skipped += 1
            if debug:
                print(f"  [skip] {doc_name}: damaged_text too short "
                      f"({len(damaged_text)} < {inference.MIN_TEXT_LEN})")
            continue
        if len(damaged_text) >= inference.TEXT_LEN:
            n_windows_skipped += 1
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
                temperature=temperature,
                unk_restoration_max_len=unk_max_len,
            )
        except Exception as e:
            n_failures += 1
            print(f"  restore() failed on {doc_name}: {e}", file=sys.stderr)
            continue

        top_preds, candidates = extract_predictions_for_damages(
            restoration, dmg_records, setting)

        results_for_this_window = []
        for rec, top_pred, cand_list in zip(dmg_records, top_preds, candidates):
            r = PredictionResult(
                original_text=text,
                masked_text=damaged_text,
                predicted_sequence=top_pred,
                predicted_sequences=cand_list[:topn],
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
        if not debug and n_windows_used % 50 == 0:
            elapsed = time.time() - t0
            rate = n_windows_used / max(elapsed, 1e-6)
            print(f"  [{setting}] {n_windows_used} windows / "
                  f"{n_damages_total} damages in {elapsed:.1f}s "
                  f"({rate:.2f} win/s)")

    aggregate = aggregate_metrics(results, topn=topn)
    aggregate.update({
        'n_windows_used': n_windows_used,
        'n_windows_skipped': n_windows_skipped,
        'n_failures': n_failures,
        'n_damages_total': n_damages_total,
        'elapsed_s': time.time() - t0,
    })
    return results, aggregate


def _print_report(setting, agg, topn):
    print(f"\n=== Setting: {setting} ===")
    print(f"windows used:     {agg['n_windows_used']}")
    print(f"windows skipped:  {agg['n_windows_skipped']}")
    print(f"restore failures: {agg['n_failures']}")
    print(f"damages scored:   {agg['n_damages_total']}")
    print(f"elapsed:          {agg['elapsed_s']:.1f}s")
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
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True,
                   help='Path to a flat checkpoint. Run '
                        'convert_snapshot_to_flat.py first if you only have '
                        'a jaxline snapshot.')
    p.add_argument('--windows', required=True,
                   help='aeneas_test_windows.json')
    p.add_argument('--damages', required=True,
                   help='damage_spans_aeneas.json')
    p.add_argument('--setting', choices=['known', 'unknown', 'both'],
                   default='both')
    p.add_argument('--topn', type=int, default=5)
    p.add_argument('--beam-width', type=int,
                   default=inference.RESTORATION_BEAM_WIDTH)
    p.add_argument('--unk-max-len', type=int,
                   default=inference.UNK_RESTORATION_MAX_LEN)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--out-prefix', default='results')
    p.add_argument('--max-windows', type=int, default=None,
                   help='Stop after this many windows; auto-enables debug '
                        'output.')
    p.add_argument('--language', default='latin', choices=['latin', 'greek'])
    args = p.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    model_config, region_map, alphabet, params, forward = load_checkpoint(
        args.checkpoint, language=args.language)
    vocab_char_size = model_config['vocab_char_size']
    print(f"  vocab_char_size={vocab_char_size}  "
          f"emb_dim={model_config.get('emb_dim')}")

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
            debug=debug, temperature=args.temperature,
        )
        _print_report(setting, aggregate, args.topn)
        out_path = f"{args.out_prefix}_{setting}.json"
        _dump_results(out_path, results, aggregate)
        print(f"\nWrote {out_path}  ({len(results)} damages, "
              f"{os.path.getsize(out_path)} bytes)")


if __name__ == '__main__':
    main()