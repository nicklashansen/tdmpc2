"""
Compare a released TD-MPC2 checkpoint on the CPU and on an accelerator.

Needs no simulator. The observation size, the action size and (for the multi-task
models) the task list are read from the checkpoint. Observations are random, the
planning noise is drawn once on the CPU and reused on both devices.

Example:
	python tests/check_checkpoint_parity.py /path/to/dog-run-1.pt --device xpu
	python tests/check_checkpoint_parity.py /path/to/mt30-5M.pt --device xpu --task-index 3
"""
import argparse

import torch

import helpers


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument('checkpoint')
	parser.add_argument('--device', default=None, help='device to compare with the CPU (default: detect)')
	parser.add_argument('--model-size', type=int, default=5)
	parser.add_argument('--episode-length', type=int, default=500)
	parser.add_argument('--task-index', type=int, default=0, help='task of a multi-task checkpoint')
	parser.add_argument('--batch', type=int, default=4)
	parser.add_argument('--compile', action='store_true', help='torch.compile the accelerator agent')
	parser.add_argument('--atol', type=float, default=1e-3)
	parser.add_argument('--rtol', type=float, default=1e-3)
	args = parser.parse_args()

	from common import device as dev
	device = dev.resolve_device(args.device)
	assert device.type != 'cpu', 'No accelerator to compare with; pass --device xpu or --device cuda.'

	state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
	meta = state.get('metadata', {})
	weights = state['model'] if 'model' in state else state
	multitask = '_task_emb.weight' in weights
	obs_dim = weights['_encoder.state.0.weight'].shape[1]
	action_dim = weights['_pi.2.weight'].shape[0] // 2
	task_dim = weights['_task_emb.weight'].shape[1] if multitask else 0
	if multitask:
		obs_dim -= task_dim
	tasks = meta.get('tasks') if multitask else None
	print(f'obs_dim={obs_dim} action_dim={action_dim} multitask={multitask} task_dim={task_dim}')

	def build(device_name, compile_):
		cfg = helpers.make_cfg(
			obs_dim=obs_dim, action_dim=action_dim, episode_length=args.episode_length,
			model_size=args.model_size, device=device_name, tasks=tasks, task_dim=task_dim,
			horizon=3, num_samples=512, num_elites=64, num_pi_trajs=24, iterations=6,
			compile=compile_,
		)
		agent = helpers.build_agent(cfg)
		agent.load(args.checkpoint)
		return agent

	cpu_agent = build('cpu', False)
	acc_agent = build(str(device), args.compile)
	task = args.task_index if multitask else None
	gen = torch.Generator().manual_seed(1)
	obs = torch.randn(args.batch, obs_dim, generator=gen)
	ref = helpers.run_agent(cpu_agent, obs, task=task)
	got = helpers.run_agent(acc_agent, obs, task=task)
	print(f'device={device} torch={torch.__version__} compile={args.compile}')
	print(f'{"tensor":<14}{"mse":>12}{"rel_mse":>12}{"max_abs":>12}{"inside":>9}   (atol={args.atol}, rtol={args.rtol})')
	for key, row in helpers.compare(ref, got, args.atol, args.rtol).items():
		print(f'{key:<14}{row["mse"]:>12.3e}{row["rel_mse"]:>12.3e}{row["max_abs"]:>12.3e}{row["inside"]*100:>8.1f}%')


if __name__ == '__main__':
	main()
