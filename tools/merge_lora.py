#!/usr/bin/env python3
"""Small tool: merge LoRA adapters into a checkpoint and write a new file.

Behavior:
- load checkpoint (tries strict load, then simple 'module.' prefix mapping)
- instantiate `TestModel`, load weights, call `model.merge_lora()`
- save merged `state_dict` to `<input>_merged.pth` (or `--out`)
"""
import argparse, os, sys
import torch

# find repo root by locating Network.py (fallback to parent of tools/)
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = script_dir
cur = script_dir
while True:
	candidate = os.path.join(cur, 'Network.py')
	if os.path.exists(candidate):
		repo_root = cur
		break
	parent = os.path.dirname(cur)
	if parent == cur:
		repo_root = os.path.abspath(os.path.join(script_dir, '..'))
		break
	cur = parent
if repo_root not in sys.path:
	sys.path.insert(0, repo_root)

from Network import TestModel


def load_state_compat(state, model):
	"""Try strict load; otherwise map keys with/without 'module.' prefix."""
	try:
		model.load_state_dict(state)
		return True
	except Exception:
		model_keys = set(model.state_dict().keys())
		mapped = {}
		for k, v in state.items():
			if k in model_keys:
				mapped[k] = v
			elif k.startswith('module.') and k[len('module.'): ] in model_keys:
				mapped[k[len('module.'):]] = v
			elif ('module.' + k) in model_keys:
				mapped['module.' + k] = v
		if not mapped:
			return False
		merged = model.state_dict()
		merged.update(mapped)
		try:
			model.load_state_dict(merged)
			return True
		except Exception:
			return False


def merge_checkpoint(checkpoint_path, out_path=None, device='cpu', N=192, M=320, num_slices=5):
	if not os.path.exists(checkpoint_path):
		raise FileNotFoundError(checkpoint_path)

	ckpt = torch.load(checkpoint_path, map_location=torch.device(device))
	state = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt

	model = TestModel(N=N, M=M, num_slices=num_slices, flag_inference=False).to(device)
	if not load_state_compat(state, model):
		raise RuntimeError('Failed to load checkpoint into model')

	model.merge_lora()

	# default output: <input>_merged(.pth)
	if out_path is None:
		base, ext = os.path.splitext(checkpoint_path)
		out_path = base + '_merged' + (ext if ext else '.pth')
	elif os.path.isdir(out_path):
		base_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
		out_path = os.path.join(out_path, base_name + '_merged.pth')
	else:
		_base, _ext = os.path.splitext(out_path)
		if _ext == '':
			out_path = out_path + '.pth'

	state_out = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
	if isinstance(ckpt, dict) and 'state_dict' in ckpt:
		new_ckpt = dict(ckpt)
		new_ckpt['state_dict'] = state_out
		torch.save(new_ckpt, out_path)
	else:
		torch.save({'state_dict': state_out}, out_path)

	return out_path


def main():
	p = argparse.ArgumentParser(description='Merge LoRA adapters into checkpoint')
	p.add_argument('--checkpoint', '-c', required=True)
	p.add_argument('--out', '-o', default=None)
	p.add_argument('--device', '-d', default='cpu')
	p.add_argument('--N', type=int, default=192)
	p.add_argument('--M', type=int, default=320)
	p.add_argument('--num_slices', type=int, default=5)
	args = p.parse_args()

	out = merge_checkpoint(args.checkpoint, out_path=args.out, device=args.device,
						   N=args.N, M=args.M, num_slices=args.num_slices)
	print(f'Merged checkpoint written: {out}')


if __name__ == '__main__':
	main()


