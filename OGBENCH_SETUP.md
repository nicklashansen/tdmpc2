# OGBench Integration for TD-MPC2

This guide explains how to train TD-MPC2 on OGBench (Open-World Generalist Benchmark) tasks.

## Installation

### 1. Install OGBench

First, you need to install the OGBench package. The installation method depends on which OGBench variant you're using:

#### Option A: If using the official OGBench
```bash
conda activate tdmpc2
pip install ogbench
```

OGBench requires `mujoco >= 3.1.6`, `dm_control >= 1.0.20`, and `gymnasium`.

#### Option B: If using a custom or forked version
```bash
conda activate tdmpc2
# Clone the repository
git clone https://github.com/[org]/ogbench.git
cd ogbench
pip install -e .
```

### 2. Verify Installation

Test that OGBench is properly installed:
```bash
conda activate tdmpc2
python -c "import ogbench; print('OGBench successfully installed!')"
```

## Usage

### Training on OGBench Tasks

The integration supports OGBench dataset names with the `ogb-` prefix. Use the OGBench dataset name directly after the prefix. For example:

```bash
cd tdmpc2
conda activate tdmpc2

# Train on a specific OGBench dataset
python train.py task=ogb-humanoidmaze-large-navigate-v0 model_size=5 steps=1000000

# Train with different model sizes
python train.py task=ogb-antmaze-large-navigate-v0 model_size=19 steps=2000000

# Train on a standard offline RL dataset (singletask)
python train.py task=ogb-cube-double-play-singletask-v0 model_size=5
```

### Available Tasks

Use the dataset names listed in the OGBench README (for example: `humanoidmaze-large-navigate-v0`, `antmaze-large-navigate-v0`, `cube-double-play-singletask-v0`, `scene-play-singletask-v0`).

Prefix any dataset name with `ogb-` to route it through the OGBench wrapper, e.g. `ogb-humanoidmaze-large-navigate-v0`.

### Configuration

You can customize training parameters in the command line:

```bash
# Custom hyperparameters
python train.py \
    task=ogb-humanoidmaze-large-navigate-v0 \
    model_size=5 \
    steps=5000000 \
    batch_size=256 \
    lr=3e-4 \
    seed=42 \
    wandb_project=ogbench-experiments

# Disable video saving for faster training
python train.py \
    task=ogb-humanoidmaze-large-navigate-v0 \
    model_size=5 \
    save_video=false

# Custom evaluation frequency
python train.py \
    task=ogb-humanoidmaze-large-navigate-v0 \
    model_size=5 \
    eval_freq=25000
```

### Evaluation

To evaluate a trained model:

```bash
python evaluate.py \
    checkpoint=<path/to/checkpoint.pt> \
    task=ogb-humanoidmaze-large-navigate-v0 \
    eval_episodes=20
```

### Multi-Task Training (Advanced)

For multi-task training on multiple OGBench tasks:

```bash
python train.py \
    task=mt-ogbench \
    tasks=[ogb-humanoidmaze-large-navigate-v0,ogb-antmaze-large-navigate-v0,ogb-cube-double-play-singletask-v0] \
    model_size=19 \
    steps=10000000
```

## Customization

### Adding New OGBench Tasks

No task list changes are required. Just use the dataset name from OGBench and prefix it with `ogb-`.
If you want shorthand aliases, you can add them to `OGBENCH_TASKS` in `tdmpc2/envs/ogbench.py`.

### Adjusting Episode Length

Different tasks may require different episode lengths. You can adjust this in the wrapper or via command line:

```python
# In ogbench.py, modify the max_steps calculation
max_steps = {
    'ogb-humanoidmaze-large-navigate-v0': 1000,
    'ogb-antmaze-large-navigate-v0': 1000,
    # Add more task-specific limits
}.get(cfg.task, 200)  # Default to 200
```

## Troubleshooting

### Issue: OGBench not found
**Solution**: Make sure OGBench is installed in the correct conda environment:
```bash
conda activate tdmpc2
pip install ogbench
```

### Issue: Environment creation fails
**Solution**: Check that the dataset name is correct and that OGBench can load it:
```bash
python -c "import ogbench; ogbench.make_env_and_datasets('humanoidmaze-large-navigate-v0', env_only=True)"
```

### Issue: CUDA/GPU errors
**Solution**: Ensure CUDA is properly configured:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Task-Specific Notes

- **State Observations**: OGBench supports both state and pixel observations, but this TD-MPC2 wrapper currently expects state observations
- **Frame Skip**: The wrapper uses frame skip of 2 by default (consistent with other TD-MPC2 environments)
- **Render Mode**: Rendering is set to 'rgb_array' for video generation

## Performance Tips

1. **Model Size**: Start with `model_size=5` for faster experimentation, use larger models (19, 48, 317) for better performance
2. **Batch Size**: Increase batch size if you have more GPU memory
3. **Learning Rate**: The default `lr=3e-4` works well, but you may need to tune it for specific tasks
4. **Evaluation**: Set `eval_freq` appropriately - more frequent evaluation slows training but provides better monitoring

## Example Training Commands

### Quick test run
```bash
python train.py task=ogb-humanoidmaze-large-navigate-v0 model_size=1 steps=100000 save_video=false
```

### Standard training
```bash
python train.py task=ogb-humanoidmaze-large-navigate-v0 model_size=5 steps=1000000
```

### High-performance training
```bash
python train.py task=ogb-humanoidmaze-large-navigate-v0 model_size=48 steps=5000000 batch_size=512
```

### With WandB logging
```bash
python train.py \
    task=ogb-humanoidmaze-large-navigate-v0 \
    model_size=5 \
    wandb_project=my-ogbench-experiments \
    wandb_entity=my-username \
    enable_wandb=true
```

## References

- TD-MPC2 Paper: https://arxiv.org/abs/2310.16828
- TD-MPC2 Website: https://www.tdmpc2.com
- OGBench: [Add link to OGBench repository/paper]
