
# Latent Dynamics Diffusion MPC

An advanced diffusion-based model predictive control framework for continuous control problems, extending the TD-MPC2 architecture with diffusion models for robust and scalable reinforcement learning.

## Overview

This repository implements a novel diffusion-based approach to Model Predictive Control using learned latent dynamics models. By leveraging denoising diffusion probabilistic models for both dynamics modeling and trajectory optimization, this framework achieves robust performance across a variety of continuous control tasks.

Key features:
- Diffusion-based dynamics modeling in latent space for multi-scale temporal dynamics
- Sample-efficient trajectory optimization through iterative denoising
- Transformer-based architecture for complex spatiotemporal dependencies
- Task-conditioned multi-task learning across diverse environments
- Support for both state and image observations
- Comprehensive benchmarking across MetaWorld, DMControl, ManiSkill2, and MyoSuite environments

## Installation

### Docker Installation (Recommended)

```bash
# Build the docker image
cd docker && docker build . -t ameyj/tdmpc2:1.0.0

# Run the container with GPU support and repository mounted
sudo docker run -i -v ~/robot_learning/tdmpc2:/tdmpc2 --gpus all -t ameyj/tdmpc2:1.0.0 /bin/bash
```

### Manual Installation

```bash
# Clone the repository
git clone -b testing/TD-MPC2/Isaac https://github.com/ameyj17/tdmpc2.git
cd tdmpc2

# Create conda environment
conda env create -f docker/environment.yaml
pip install gym==0.21.0

# For ManiSkill2 support
python -m mani_skill2.utils.download_asset all
export MS2_ASSET_DIR=/path/to/assets
```

## Usage

### Training on MetaWorld Tasks

```bash
# Train on door-open task with 19M parameter model
python train_mw10.py task=mw-door-open model_size=19 batch_size=256 buffer_size=50000 steps=1000000

# Train on other MetaWorld tasks
python train_mw10.py task=mw-assembly model_size=5 batch_size=256 buffer_size=50000 steps=500000
```

### Training on Various Environments

```bash
# Train on DMControl tasks
python train.py task=dog-run steps=7000000 compile=true

# Train with image observations
python train.py task=walker-walk obs=rgb model_size=19

# Multi-task training
python train.py task=mt30 model_size=48 batch_size=1024
```

### Evaluation

```bash
# Evaluate a trained model
python evaluate.py task=mw-door-open checkpoint=/path/to/checkpoint.pt save_video=true

# Evaluate multi-task models
python evaluate.py task=mt80 model_size=48 checkpoint=/path/to/mt80-48M.pt
```

## Model Architecture

The diffusion-based latent dynamics model combines several key components:

- **Transformer-based Diffusion Dynamics**: Learns latent space transitions using a denoising diffusion process
- **Diffusion Action Proposal Network**: Generates optimized action sequences through noise prediction
- **Multi-scale Dynamics Modeling**: Captures dynamics at different temporal resolutions
- **Task-Conditioned Architecture**: Enables transfer between tasks with task-specific embeddings

Our implementation extends TD-MPC2 with specialized diffusion models to better handle complex temporal dependencies and improve generalization across tasks.

## Supported Environments

- **DMControl**: 39 tasks including custom extensions
- **MetaWorld**: 50 diverse manipulation tasks with challenging dynamics
- **ManiSkill2**: Advanced robotic manipulation scenarios
- **MyoSuite**: Musculoskeletal control tasks with complex biomechanics

## Performance Enhancements

The testing/TD-MPC2/Isaac branch features several optimizations:

- **4.5x Faster Training**: With torch.compile enabled via `compile=true`
- **Multi-GPU Support**: Distributed training for large-scale experiments
- **Memory Optimizations**: Reduced memory footprint for efficient multi-task learning
- **Enhanced Diffusion Models**: Improved sampling efficiency and dynamics prediction

## License

MIT License

## Citation

If using this codebase, please cite:
```
@inproceedings{hansen2024tdmpc2,
  title={TD-MPC2: Scalable, Robust World Models for Continuous Control}, 
  author={Nicklas Hansen and Hao Su and Xiaolong Wang},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

This implementation builds upon TD-MPC2 with significant extensions for diffusion-based latent dynamics modeling.
