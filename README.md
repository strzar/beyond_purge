# Beyond Binary Rewards: A Comparative Study of Reward Design for Reinforcement Unlearning

## Abstract

Machine unlearning seeks to selectively remove specific knowledge from trained
language models without full retraining, a growing necessity under privacy
regulations such as GDPR and the EU AI Act. Recent work has reformulated
unlearning as a Reinforcement Learning with Verifiable Rewards (RLVR) problem,
where models are optimized against verifiable rewards computed directly from
their outputs. However, existing methods rely on sparse binary rewards that
provide minimal learning signal — a completion either avoids forbidden content
or it does not — limiting convergence speed. In this paper, we study how
reward design affects unlearning efficiency within the Reinforcement
Unlearning (RUL) framework. We introduce a principled reward decomposition
framework that decouples verifiability from sparsity, and propose two new
reward functions: an exponential reward that provides graded penalties based
on the count of forbidden-concept occurrences, and a PageRank-inspired reward
that weights penalties by semantic importance. We conduct experiments on the
Real World Knowledge Unlearning (RWKU) benchmark, demonstrating that both
rewards consistently outperform the binary setting, while reaching similar
forgetting performance up to 3× faster and preserving general model utility.
Our results show that reward design is a key driver of unlearning efficiency,
offering a practical path toward scalable and efficient machine unlearning.

**Keywords:** Machine Unlearning · Reinforcement Learning · RLVR · GRPO ·
Large Language Models

This repository implements a GRPO-based unlearning pipeline for Large
Language Models using modular reward functions:
- `binary`: strict hit/no-hit penalty on forget terms
- `exponential_decay`: smooth penalty based on number of forget-term matches
- `pagerank`: graph-weighted penalty using semantic importance of forget terms

## Repository at a glance

- `src/purge.py`: main training entry point (Hydra + TRL GRPOTrainer)
- `src/configs/`: Hydra config groups (`entity`, `model`, `training`, `reward`, `diagnostics`, `evaluation`, `paths`)
- `src/rewards/`: reward implementations (`binary`, `exponential_decay`, `pagerank`)
- `src/diagnostics/`: training-time diagnostics callbacks and post-training diagnostic bundles
- `src/evaluation/`: RWKU benchmark evaluation callback
- `src/minimal.py`: lightweight non-Hydra prototype script for fast experimentation
- `data/PURGE/<entity>/`: per-entity unlearning data (`qa_pairs.json`, `fts.json`)
- `data/RWKU/<entity>/`: per-entity RWKU benchmark evaluation data
- `scripts/run_h200.sh`, `scripts/run_a100.sh`: SLURM batch runs over selected entities
- `src/misc/`: helper scripts for data generation, token budgeting, and HF upload

## Environment setup

1. Clone and enter repo

```bash
git clone <this-repository-url>
cd purge
```

2. Create environment (Python 3.10+ recommended)

```bash
conda create -n purge
conda activate purge
pip install -r requirements.txt
```

3. Authenticate (if needed)

```bash
huggingface-cli login
# optional
wandb login
```

## Quick start

Important: current default path configs are authored for running from `scripts/`.

```bash
cd scripts
python ../src/purge.py training=fast entity=1_Stephen_King
```

That command runs a short debug job using:
- model: `microsoft/Phi-3-mini-4k-instruct`
- reward: `exponential_decay`
- dataset subset: 20 samples (`training=fast`)

## Running experiments

### Single entity

From `scripts/`:

```bash
python ../src/purge.py entity=74_Socrates
```

### Change reward function

```bash
python ../src/purge.py entity=74_Socrates reward=binary training=binary
python ../src/purge.py entity=74_Socrates reward=exponential_decay training=exponential_decay
python ../src/purge.py entity=74_Socrates reward=pagerank training=pagerank
```

### Change model

```bash
python ../src/purge.py model=qwen2.5-1.5b
python ../src/purge.py model=qwen2.5-3b
python ../src/purge.py model=llama3.2-1b
```

### Full-data training

```bash
python ../src/purge.py training=full
```

### Hydra sweeps (multi-run)

```bash
python ../src/purge.py --multirun \
  entity=1_Stephen_King,2_Confucius \
  reward=binary,exponential_decay
```

## Configuration system (Hydra)

Main config: `src/configs/config.yaml`

Default groups:
- `entity`: one target identity (files in `src/configs/entity/`)
- `model`: base model selection (files in `src/configs/model/`)
- `training`: optimizer/runtime knobs (files in `src/configs/training/`)
- `reward`: reward type parameters (files in `src/configs/reward/`)
- `diagnostics`: live/post-training diagnostics settings (files in `src/configs/diagnostics/`)
- `evaluation`: RWKU benchmark evaluation settings (files in `src/configs/evaluation/`)
- `paths`: dataset/model output paths (files in `src/configs/paths/`)

Examples:

```bash
python ../src/purge.py \
  entity=24_Beyoncé \
  training.max_steps=1500 \
  training.per_device_train_batch_size=1 \
  training.gradient_accumulation_steps=16
```

## Data format

Each entity folder is expected at:

`data/PURGE/<entity_target>/`

Required files:
- `fts.json`: list of forget terms/entities
- `qa_pairs.json`: list of prompt-response objects used for GRPO training

`qa_pairs.json` example:

```json
[
  {"prompt": "...", "response": "..."},
  {"prompt": "...", "response": "..."}
]
```

RWKU benchmark evaluation data (used by `evaluation: default`) is expected at
`data/RWKU/<entity_target>/`.

## Cluster usage (SLURM)

### H200 batch script

```bash
cd scripts
sbatch run_h200.sh
```

- Edit `names=(...)` in `scripts/run_h200.sh` to choose entity subset.
- Logs are written to `logs/`.
- Uncomment and set `--nodelist` if your cluster requires pinning to a specific node.

### A100 script

```bash
cd scripts
sbatch run_a100.sh
```

## Outputs and logging

By default, trained checkpoints are written under:

`models/<reward.type>/<model-name>-<entity>-<reward-type>`

Hydra run metadata is written under timestamped `outputs/` and `multirun/` directories.

If `wandb` is enabled, runs are tracked automatically. To disable online sync:

```bash
export WANDB_MODE=offline
```

## Utility scripts

- `src/minimal.py`: lightweight non-Hydra prototype script for fast experimentation and integration
- `src/misc/data_generation/generate_responses.py`: build `qa_pairs.json` from target prompts
- `src/misc/data_generation/generate_forget_words.py`: create manual NER prompt files
- `src/misc/huggingface_uploads/upload.py`: upload trained checkpoints (`--target <entity> --hf-org <your-hf-org>`, edit placeholder paths first)
- `scripts/upload.sh`: batch upload loop over entities (set `HF_ORG=<your-hf-org>` before running)

## Troubleshooting

- `ModuleNotFoundError: No module named 'trl'`
  - Install dependencies with `pip install -r requirements.txt` in the active environment.

- `FileNotFoundError` for `../data/PURGE/...`
  - Run training from `scripts/` as shown above, or override `paths.*` in Hydra.

- CUDA OOM / slow runs
  - Reduce `training.per_device_train_batch_size`, `training.num_generations`, or switch to `training=fast`.


## License

This project is licensed under the MIT License.
