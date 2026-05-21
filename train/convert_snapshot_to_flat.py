"""Convert a jaxline DiskCheckpointer snapshot to the flat (released) format.

Input:  checkpoint_latest_id0000.pkl   (jaxline snapshot: [SnapshotNT(id, pickle_nest)])
Output: <name>_flat.pkl                (flat: {model_config, params, region_map, alphabet})

The flat schema is exactly what inference_example.load_checkpoint expects:
  - 'model_config': dict of Model() kwargs
  - 'params':       dict, {'params': <tree>, 'params_axes': <metadata>}
  - 'region_map':   dict (we copy from the released baseline checkpoint)
  - 'alphabet':     dict of alphabet vocabulary entries (we copy too)

Why we copy 'region_map' and 'alphabet' from the baseline:
  These are non-trainable metadata. The fine-tune doesn't modify them, and
  they're identical across the released checkpoint and any fine-tune of it.
  The snapshot doesn't include them (jaxline checkpoints only persist
  CHECKPOINT_ATTRS, which here means '_params' and '_opt_state'). The
  cleanest path is to take them from the released baseline.

  region_map is only used by inference.contextualize(); restore() and
  attribute() don't need it. We carry it anyway so the output is a complete
  drop-in replacement for the baseline pickle.

Usage:
  python convert_snapshot_to_flat.py \
      --snapshot /path/to/finetuned_seedN/checkpoint_latest_id0000.pkl \
      --baseline /path/to/aeneas_117149994_2.pkl \
      --out      /path/to/finetuned_seedN_flat.pkl
"""

import argparse
import pickle


def _extract_snapshot_params(snapshot_list):
    """Pull the wrapped params dict ({params, params_axes}) out of a snapshot.

    Does NOT unwrap, strip device axes, or transform in any other way. The
    snapshot stores exactly the variables dict that model.init returns
    (which has both `params` and `params_axes` keys), and that's exactly
    what the released checkpoint stores too, so we pass it through.
    """
    if not isinstance(snapshot_list, list) or not snapshot_list:
        raise ValueError("Snapshot file must contain a non-empty list of "
                         f"SnapshotNT; got {type(snapshot_list).__name__}.")
    snap = snapshot_list[-1]
    if not (hasattr(snap, '_fields') and 'pickle_nest' in snap._fields):
        raise ValueError("Snapshot entries must be SnapshotNT-shaped; "
                         f"got {type(snap).__name__}.")
    nest = snap.pickle_nest
    if hasattr(nest, 'to_dict'):
        nest = nest.to_dict()
    exp_state = nest.get('experiment_module', {})
    if hasattr(exp_state, 'to_dict'):
        exp_state = exp_state.to_dict()
    params = exp_state.get('_params') or exp_state.get('params')
    if params is None:
        raise KeyError(
            "Snapshot's experiment_module has no '_params' or 'params'. "
            f"Keys present: {list(exp_state.keys())}")
    return params, snap.id


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--snapshot', required=True,
                   help='Path to the jaxline snapshot (e.g. '
                        'finetuned_seedN/checkpoint_latest_id0000.pkl).')
    p.add_argument('--baseline', required=True,
                   help='Path to the released flat checkpoint (e.g. '
                        'aeneas_117149994_2.pkl). Used to copy '
                        'model_config, region_map, and alphabet.')
    p.add_argument('--out', required=True,
                   help='Output flat-format pickle path.')
    args = p.parse_args()

    print(f"Loading baseline: {args.baseline}")
    with open(args.baseline, 'rb') as f:
        baseline = pickle.load(f)
    for key in ('model_config', 'region_map', 'alphabet'):
        if key not in baseline:
            raise KeyError(f"Baseline {args.baseline} missing key {key!r}; "
                           f"keys present: {list(baseline.keys())}")
    print(f"  baseline keys: {list(baseline.keys())}")

    print(f"Loading snapshot: {args.snapshot}")
    with open(args.snapshot, 'rb') as f:
        snapshot = pickle.load(f)
    params, snap_id = _extract_snapshot_params(snapshot)
    print(f"  snapshot id={snap_id}")
    if hasattr(params, 'keys'):
        print(f"  params top-level keys: {list(params.keys())}")

    # Build the flat pickle.
    flat = {
        'model_config': baseline['model_config'],
        'params': params,
        'region_map': baseline['region_map'],
        'alphabet': baseline['alphabet'],
    }

    print(f"Writing flat checkpoint: {args.out}")
    with open(args.out, 'wb') as f:
        pickle.dump(flat, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done.")


if __name__ == '__main__':
    main()