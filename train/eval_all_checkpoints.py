# Copyright 2025 the Aeneas Authors
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Evaluate every saved checkpoint on the dev set.

The continuous `--jaxline_mode=eval` loop only ever evaluates the *latest*
checkpoint. This script walks every permanent per-snapshot file written by
DiskCheckpointer --

    <checkpoint_dir>/checkpoint_<series>_id<NNNN>.pkl

-- loads the params straight out of each one, runs the dev-set evaluation
using experiment.py's own _eval_epoch (so the metric is identical to the
jaxline eval loop / training logs), and prints a table of
(id, global_step, score, mask_acc) plus the winner.

It does NOT use jaxline's checkpointer / GLOBAL_CHECKPOINT_DICT machinery:
it reads the pickle, pulls 'params' out, replicates across devices,
assigns to the Experiment, and calls evaluate().

Usage (inside the conda env, on a GPU node, from the train/ dir):

    python eval_all_checkpoints.py --config=config_paleo_eval.py --logtostderr

Optional:
    --series=latest
    --csv_out=dev_scores.csv
    --inspect            # print the structure of the first pickle and exit
"""

import csv
import glob
import os
import pickle
import re

from absl import app
from absl import flags
from absl import logging
import jax
import numpy as np
from jaxline import utils as jl_utils

import experiment as experiment_lib


FLAGS = flags.FLAGS
# NOTE: do not define a '--config' flag here. Importing `experiment` pulls in
# jaxline.platform, which already registers '--config' (a config_flags
# config_file flag). Redefining it raises DuplicateFlagError. We simply read
# FLAGS.config, which jaxline has already set up.
flags.DEFINE_string('series', 'latest', 'Checkpoint series name.')
flags.DEFINE_string('csv_out', '', 'Optional path to also write a CSV.')
flags.DEFINE_bool('inspect', False,
                  'Print the structure of the first checkpoint and exit.')


def _find_checkpoint_files(checkpoint_dir, series):
  """Returns [(id, path), ...] sorted by id for all per-snapshot files."""
  pattern = os.path.join(checkpoint_dir, f'checkpoint_{series}_id*.pkl')
  id_re = re.compile(rf'checkpoint_{re.escape(series)}_id(\d+)\.pkl$')
  found = []
  for path in glob.glob(pattern):
    m = id_re.search(os.path.basename(path))
    if m:
      found.append((int(m.group(1)), path))
  found.sort(key=lambda t: t[0])
  return found


def _describe(obj, depth=0, max_depth=5):
  """Recursively print the shape of a nested structure (for --inspect)."""
  pad = '  ' * depth
  if depth > max_depth:
    print(f'{pad}...')
    return
  if hasattr(obj, 'to_dict'):
    obj = obj.to_dict()
  if isinstance(obj, dict):
    for k in list(obj.keys()):
      print(f'{pad}{k!r}:')
      _describe(obj[k], depth + 1, max_depth)
  elif isinstance(obj, (list, tuple)):
    fields = getattr(obj, '_fields', None)
    label = f'namedtuple{fields}' if fields else type(obj).__name__
    print(f'{pad}{label} len={len(obj)}')
    if obj:
      _describe(obj[0], depth + 1, max_depth)
  elif hasattr(obj, 'shape'):
    print(f'{pad}array shape={obj.shape} dtype={obj.dtype}')
  else:
    print(f'{pad}{type(obj).__name__}')


def _extract_params_and_step(pkl_obj):
  """Pulls (params, global_step) out of a DiskCheckpointer snapshot pickle.

  DiskCheckpointer.save() pickles a list of jaxline SnapshotNT(id, pickle_nest)
  namedtuples. pickle_nest is a ConfigDict-like nest whose 'experiment_module'
  entry is AbstractExperiment.snapshot_state() -> a dict keyed by the
  CHECKPOINT_ATTRS *values*, i.e. {'params': ..., 'opt_state': ...}.

  Defensive: accepts a raw history list, a single snapshot, or an already
  flattened dict, and searches for a 'params' / '_params' key.
  """
  obj = pkl_obj
  if isinstance(obj, (list, tuple)) and obj and not hasattr(obj, '_fields'):
    obj = obj[-1]  # history list -> last snapshot

  nest = obj
  if hasattr(obj, '_fields') and 'pickle_nest' in obj._fields:
    nest = obj.pickle_nest

  if hasattr(nest, 'to_dict'):
    nest = nest.to_dict()
  if not isinstance(nest, dict):
    raise TypeError(f'Unexpected snapshot nest type: {type(nest)}')

  global_step = nest.get('global_step', None)

  exp_state = nest.get('experiment_module', nest)
  if hasattr(exp_state, 'to_dict'):
    exp_state = exp_state.to_dict()

  params = None
  if isinstance(exp_state, dict):
    for key in ('params', '_params'):
      if key in exp_state:
        params = exp_state[key]
        break

  if params is None:
    raise KeyError(
        "Could not find 'params' in checkpoint. Re-run with --inspect to "
        'see the pickle structure.')

  return params, global_step


def main(_):
  config = FLAGS.config
  exp_config = config.experiment_kwargs.config
  checkpoint_dir = config.checkpoint_dir
  series = FLAGS.series

  ckpts = _find_checkpoint_files(checkpoint_dir, series)
  if not ckpts:
    raise FileNotFoundError(
        f'No checkpoint_{series}_id*.pkl files found in {checkpoint_dir}')
  logging.info('Found %d checkpoint(s) to evaluate.', len(ckpts))

  if FLAGS.inspect:
    first_path = ckpts[0][1]
    print(f'Structure of {first_path}:')
    with open(first_path, 'rb') as f:
      _describe(pickle.load(f))
    return

  # Build one Experiment in eval mode; reused across all checkpoints.
  init_rng = jax.random.PRNGKey(exp_config.random_seed)
  exp = experiment_lib.Experiment(
      mode='eval', init_rng=init_rng, config=exp_config)

  results = []
  rng = init_rng
  ckpts=(0,'/leonardo_work/IscrC_CoIta/predictingthepast/checkpoint/aeneas_117149994_2.pkl')
  for ckpt_id, path in ckpts:
    with open(path, 'rb') as f:
      pkl_obj = pickle.load(f)
    params, global_step = _extract_params_and_step(pkl_obj)
    if global_step is None:
      global_step = ckpt_id  # fallback if step wasn't stored

    # _eval_epoch does get_first(self._params), so _params must be
    # replicated across local devices.
    exp._params = jl_utils.bcast_local_devices(params)
    exp._eval_input = None  # rebuild the dev pipeline fresh each checkpoint

    summary = exp.evaluate(
        global_step=jl_utils.bcast_local_devices(np.asarray(global_step)),
        rng=jl_utils.bcast_local_devices(rng))

    eval_lang = exp_config['dataset']['eval_language'][0]
    score = float(summary[f'{eval_lang}/score/eval'])
    mask_acc = float(summary.get(f'{eval_lang}/accuracy/mask', float('nan')))
    results.append((ckpt_id, int(global_step), score, mask_acc))
    logging.info('id=%04d  step=%d  score=%.4f  mask_acc=%.4f',
                 ckpt_id, int(global_step), score, mask_acc)

  # Report.
  print()
  print('=' * 56)
  print(f'{"id":>5}  {"step":>8}  {"score":>10}  {"mask_acc":>10}')
  print('-' * 56)
  for ckpt_id, step, score, mask_acc in results:
    print(f'{ckpt_id:5d}  {step:8d}  {score:10.4f}  {mask_acc:10.4f}')
  print('=' * 56)

  if results:
    best = max(results, key=lambda r: r[2])
    print(f'BEST: id={best[0]:04d}  step={best[1]}  score={best[2]:.4f}')
    print(f'  -> checkpoint_{series}_id{best[0]:04d}.pkl')

  if FLAGS.csv_out:
    with open(FLAGS.csv_out, 'w', newline='') as f:
      w = csv.writer(f)
      w.writerow(['id', 'global_step', 'score', 'mask_accuracy'])
      w.writerows(results)
    print(f'Wrote {FLAGS.csv_out}')


if __name__ == '__main__':
  # '--config' is already defined and required by jaxline.platform (imported
  # via `experiment`); we do not re-declare or re-mark it here.
  app.run(main)