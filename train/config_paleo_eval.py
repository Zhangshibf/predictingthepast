# Copyright 2025 the Aeneas Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Config for EVALUATING a fine-tuned Aeneas checkpoint on the dev split.

This is a sibling to `config_paleo.py`. The training config is left
untouched; this file is used only when launching with
`--jaxline_mode=eval`.

Differences vs. `config_paleo.py`:
  * `evaluation.batch_size` raised from 1 to 16 -- eval is batch-size
    insensitive, so this just makes it ~16x faster.
  * `evaluation.mode='valid'` (already was) -- the dataloader filters
    samples to those with `id % 10 == 4` in this mode, so we evaluate
    on dev only.
  * `repeat_eval=1` -- one pass over the 604 dev records is enough.
  * All training-time data augmentation is set to its eval-safe value
    (no random word/char/sentence mutations, no random context length).
    This makes eval results reproducible and comparable across runs.
  * `char_mask_rate_max` is left at 0.75 because at eval time the
    dataloader uses fixed `span_mask_eval_len=10` masking anyway -- the
    rate knob has no effect in `mode='valid'`.
  * `pretrained_checkpoint_path` is removed: in eval mode jaxline loads
    the latest checkpoint from `checkpoint_dir`, so we point that at
    `/finetuned/`. To eval the released baseline instead, point
    `checkpoint_dir` at a directory containing only the released pkl
    renamed to match the jaxline naming, or use the standalone
    `inference_example.py` for one-off baseline checks.
  * Optimizer block is kept (jaxline checks it exists) but is irrelevant
    in eval mode.

Launch with:
  python experiment.py --config=config_paleo_eval.py \
      --jaxline_mode=eval --logtostderr
"""

from jaxline import base_config
from ml_collections import config_dict


def get_config():
  """Return config object for fine-tuning."""

  config = base_config.get_base_config()

  # ---------------------------------------------------------------------------
  # Distributed setup. Adjust to match your hardware.
  # ---------------------------------------------------------------------------
  local_batch_size = 32
  num_devices = 1
  config.train_batch_size = local_batch_size * num_devices

  # ---------------------------------------------------------------------------
  # Macros: these are referenced from multiple places below.
  # ---------------------------------------------------------------------------
  config.macros = config_dict.ConfigDict(
      dict(
          context_char_max=768,
          date_max=800,
          date_min=-800,
          date_interval=10,
          date_bins=160,
          vision=True,
          prepend_sos=1,
      )
  )
  cm = config.macros  # Alias.

  config.experiment_kwargs = config_dict.ConfigDict(
      dict(
          config=dict(
              random_seed=4,
              # No pretrained_checkpoint_path: in eval mode jaxline loads
              # the latest checkpoint from config.checkpoint_dir below.
              random_mode_train=config.get_ref('random_mode_train'),
              random_mode_eval=config.get_ref('random_mode_eval'),
              # ---------------------------------------------------------------
              # Optimizer: lower peak LR + shorter schedule for fine-tuning.
              # ---------------------------------------------------------------
              optimizer=dict(
                  name='lamb',
                  kwargs=dict(
                      weight_decay=0.0,
                      b2=0.999,
                  ),
                  lr_schedule_kwargs=dict(
                      peak_value=1e-4,      # was 3e-3 (pretraining)
                      init_value=1e-7,
                      warmup_steps=500,     # was 4_000
                      decay_steps=15_000,   # was 1_000_000
                      end_value=1e-6,
                  ),
                  clip_adaptive=False,
                  clip_level=0.0,
              ),
              training=dict(
                  batch_size=config.get_oneway_ref('train_batch_size')
              ),
              alphabet=dict(),
              # ---------------------------------------------------------------
              # Dataset.
              # ---------------------------------------------------------------
              dataset=dict(
                  vision=dict(
                      enabled=cm.get_ref('vision'),
                      path='data/img-20231005/',
                      output_size=(224, 224),
                      emb_dim=1280,
                      zoom_factor=2,
                      zoom_sampling_log=False,
                  ),
                  train_language=['latin'],
                  eval_language=['latin'],
                  latin_dataset_path='data/aeneas_train_dev.json',
                  # iphi.json must EXIST (experiment.py opens it
                  # unconditionally) but can contain an empty array `[]`
                  greek_dataset_path='data/iphi.json',
                  greek_region_path='data/iphi-region-sub.txt',
                  latin_region_path='data/led-region-sub.txt',
                  context_char_min=25,
                  context_char_max=cm.get_ref('context_char_max'),
                  # Eval-mode reproducibility: turn off all stochastic
                  # input mutations. The dataloader's mode='valid' branch
                  # ignores most of these anyway, but setting them
                  # explicitly to "off" makes the config self-documenting
                  # and immune to dataloader changes.
                  context_char_random=False,
                  char_use_guess=True,
                  char_mask_rate_min=0.0,
                  char_mask_rate_max=0.75,    # ignored in mode='valid'
                  span_mask_eval_len=10,      # this is what eval uses
                  span_mask_ratio=0.0,        # ignored in mode='valid'
                  span_mask_geometric_p=0.1,
                  inject_missing_unk_p=0.0,   # was 0.25 (training only)
                  random_sentence_swap=0.0,   # was 0.25
                  random_char_delete=0.0,
                  random_word_delete=0.0,     # was 0.2
                  random_word_swap=0.0,
                  random_word_abbr=0.0,
                  punctuation_delete=False,   # was True (training only)
                  date_min=cm.get_ref('date_min'),
                  date_max=cm.get_ref('date_max'),
                  date_interval=cm.get_ref('date_interval'),
                  date_bins=cm.get_ref('date_bins'),
                  prepend_sos=cm.get_ref('prepend_sos'),
                  max_workers=None,
                  repeat_train=-1,
                  repeat_eval=1,           # was 10; one pass over dev is enough
                  block_list=[],
                  allow_list=[],
              ),
              # ---------------------------------------------------------------
              # Model: identical to the released checkpoint's architecture.
              # DO NOT change emb_dim, num_layers, num_heads, qkv_dim,
              # mlp_dim, vocab_char_size, output_regions, output_date,
              # context_char_max, model_type, prepend_sos -- any change
              # will break checkpoint loading.
              # ---------------------------------------------------------------
              model=dict(
                  emb_word_disable=True,
                  emb_decoder_type='tied',
                  emb_init='normal',
                  emb_norm=True,
                  emb_dim=384,
                  qkv_dim=32,
                  mlp_dim=1536,
                  num_layers=16,
                  num_heads=8,
                  vocab_char_size=32,
                  output_regions=62,           # MUST match pretraining
                  output_date=cm.get_ref('date_bins'),
                  output_date_dist=True,
                  region_date_pooling='first',
                  use_output_mlp=True,
                  max_len=cm.get_ref('context_char_max'),
                  dropout_rate=0.1,
                  attention_dropout_rate=0.1,
                  use_bfloat16=False,
                  model_type='t5',
                  feature_combine_type='concat',
                  posemb_combine_type='concat',
                  activation_fn='gelu',
                  vision=cm.get_ref('vision'),
                  prepend_sos=cm.get_ref('prepend_sos'),
              ),
              # ---------------------------------------------------------------
              # Loss: only `mask` (text restoration) and `unk` (lacuna length)
              # are active. Date/region/nsp are disabled.
              # ---------------------------------------------------------------
              loss=dict(
                  date=dict(
                      enabled=False,
                      weight=0.0,
                  ),
                  region=dict(
                      enabled=False,
                      weight=0.0,
                      label_smoothing=0.1,
                  ),
                  mask=dict(
                      enabled=True,
                      weight=3.0,
                      label_smoothing=0.05,
                  ),
                  nsp=dict(
                      enabled=False,
                      weight=0.0,
                  ),
                  unk=dict(
                      enabled=True,
                      weight=1.0,
                  ),
              ),
              evaluation=dict(
                  use_jit=True,
                  batch_size=16,    # was 1; 16x speedup, no metric impact
                  mode='valid',
                  store_model_log=False,
                  store_model_log_steps=100,
              ),
          ),
      )
  )

  # ---------------------------------------------------------------------------
  # Training loop.
  # ---------------------------------------------------------------------------
  config.training_steps = 20_000        # was 1_000_000
  config.log_train_data_interval = 10
  config.save_checkpoint_interval = 200
  # We disabled date+region, so the original score formula
  # (mask_acc + region_acc - 0.01 * date_l1) reduces effectively to
  # mask_acc only (region_acc is 0/eps because region_available is always
  # False on our dataset, and date_l1 is 0 for the same reason).
  config.best_model_eval_metric = 'latin/score/eval'
  config.checkpoint_dir = '/leonardo_work/IscrC_CoIta/predictingthepast/finetuned'
  config.train_checkpoint_all_hosts = False

  config.lock()

  return config