# Robot Reinforcement Learning for CS8813

This repository contains reinforcement learning implementations for training policies that are deployed on a real robot, as part of **CS8813 at Georgia Tech**.

For convenience, we include both **TD-MPC2** (model-based) and **PPO** (model-free) within the same framework. Trained policy checkpoints (`.pt` files) are exported and deployed separately on the physical robot.

This repo is a fork of the original [TD-MPC2](https://github.com/nicklashansen/tdmpc2) codebase by Nicklas Hansen, Hao Su, and Xiaolong Wang (UC San Diego). We reuse their training infrastructure and extend it for our purposes.

For the ROS deployment side (running trained policies on a TurtleBot), see our companion repo: [turtlebot_rl](https://github.com/Tizmax/turtlebot_rl).

---

## Original Work

> **TD-MPC2: Scalable, Robust World Models for Continuous Control**
>
> Nicklas Hansen, Hao Su, Xiaolong Wang
>
> [[Paper]](https://arxiv.org/abs/2310.16828) [[Website]](https://www.tdmpc2.com) [[Original Repo]](https://github.com/nicklashansen/tdmpc2)

If you use this codebase, please cite the original authors:

```
@inproceedings{hansen2024tdmpc2,
  title={TD-MPC2: Scalable, Robust World Models for Continuous Control},
  author={Nicklas Hansen and Hao Su and Xiaolong Wang},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

---

## Getting Started

See the original [TD-MPC2 README](https://github.com/nicklashansen/tdmpc2#getting-started) for environment setup, supported tasks, and training instructions.

---

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. Note that the repository relies on third-party code, which is subject to their respective licenses.
