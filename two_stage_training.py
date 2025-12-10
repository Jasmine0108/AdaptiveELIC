# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import random
import shutil
import sys
import os
import time
import datetime 

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
# from ELICUtilis.datasets import ImageFolder

from tensorboardX import SummaryWriter
from PIL import ImageFile, Image
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True
from ELICUtilis.utilis.utilis import DelfileList, load_checkpoint
from Network import TestModel



class ImageFolder(Dataset):
    # 1. 增加 mode 參數和 split_ratio 參數 (預設 80% 訓練集)
    #    支援從 JSON 檔讀取檔名：--split-json
    def __init__(self, root, transform=None, mode='train', split_ratio=0.8, filelist_json=None):
        self.root = root
        self.transform = transform
        self.mode = mode.lower()  # 轉小寫確保判斷正確
        self.split_ratio = split_ratio

        # 定義支援的圖片格式
        valid_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

        if not os.path.exists(root):
            raise FileNotFoundError(f"Root directory not found: {root}")

        # 如果提供了 JSON 檔，優先從 JSON 讀取檔名
        all_samples = []
        if filelist_json is not None:
            try:
                import json

                with open(filelist_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                filenames = []
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, str):
                            filenames.append(item)
                        elif isinstance(item, dict) and 'filename' in item:
                            filenames.append(item['filename'])
                elif isinstance(data, dict):
                    # common keys that may hold lists of filenames
                    for key in ('filenames', 'files', 'images', 'training', 'train'):
                        if key in data and isinstance(data[key], list):
                            for item in data[key]:
                                if isinstance(item, str):
                                    filenames.append(item)
                                elif isinstance(item, dict) and 'filename' in item:
                                    filenames.append(item['filename'])
                            break
                    # fallback: collect any list-of-dicts with 'filename' fields
                    if len(filenames) == 0:
                        for v in data.values():
                            if isinstance(v, list):
                                for item in v:
                                    if isinstance(item, dict) and 'filename' in item:
                                        filenames.append(item['filename'])
                                if len(filenames) > 0:
                                    break

                # normalize and join with root if needed
                for fn in filenames:
                    if os.path.isabs(fn):
                        candidate = fn
                    else:
                        candidate = os.path.join(root, fn)
                    if os.path.exists(candidate) and candidate.lower().endswith(valid_extensions):
                        all_samples.append(candidate)
                    else:
                        # try basename only search in root
                        base = os.path.basename(fn)
                        alt = os.path.join(root, base)
                        if os.path.exists(alt) and alt.lower().endswith(valid_extensions):
                            all_samples.append(alt)
                        else:
                            # skip missing files silently but print a warning
                            print(f"Warning: listed file not found or unsupported: {fn}")
            except Exception as e:
                print(f"Failed to load filelist JSON '{filelist_json}': {e}")

        # 若 JSON 沒提供或回傳為空，就走預設的資料夾掃描
        if len(all_samples) == 0:
            all_samples = [
                os.path.join(root, f) for f in os.listdir(root)
                if f.lower().endswith(valid_extensions)
            ]

        # 排序確保 train/test 劃分在不同運行中保持一致
        all_samples.sort()

        if len(all_samples) == 0:
            raise RuntimeError(f"Found 0 files in {root}. Supported extensions are: {valid_extensions}")

        # 執行 train/test 劃分邏輯
        total_count = len(all_samples)
        train_count = math.floor(total_count * self.split_ratio)

        if self.mode == 'train':
            self.samples = all_samples[:train_count]
            print(f"Dataset initialized in TRAIN mode. Using {len(self.samples)} files.")
        elif self.mode == 'test':
            self.samples = all_samples[train_count:]
            print(f"Dataset initialized in TEST mode. Using {len(self.samples)} files.")
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'test'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        # 讀取圖片並轉為 RGB
        sample = Image.open(path).convert('RGB')

        # 你的解析度印出程式碼 (已註解掉，如果你需要開啟可以取消註解)
        # if index < 5: 
        #     print(f"[{index}] 檔名: {os.path.basename(path)} | 原始解析度: {sample.size}")

        if self.transform is not None:
            sample = self.transform(sample)
        
        # 回傳圖片
        return sample


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["y_bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"]["y"]
        )
        out["z_bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"]["z"]
        )
        out["mse_loss"] = self.mse(output["x_hat"], target) * 255 ** 2
        out["loss"] = self.lmbda * out["mse_loss"] + out["bpp_loss"]

        return out

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def resolve_trainable_params(net, params_or_names):
    """Given either a list of parameter tensors or a list of parameter names,
    return a list of parameter tensors suitable for optimizer creation.
    """
    if params_or_names is None:
        return []
    # if already tensors, return as-is
    if len(params_or_names) > 0 and not isinstance(params_or_names[0], str):
        return params_or_names
    params_dict = dict(net.named_parameters())
    resolved = []
    for name in params_or_names:
        if name in params_dict:
            resolved.append(params_dict[name])
        else:
            # skip unknown names (could be adapter names not present in this model)
            print(f"Warning: trainable parameter name not found in model: {name}")
    return resolved


def load_pretrained_compat(model, checkpoint_state_dict, verbose=True):
    """Compatibility loader: copy only matching keys and shapes from checkpoint into model.

    This handles 'module.' prefix mismatches and skips keys that do not match by name or shape.
    The function updates the model's state dict and calls the model's `load_state_dict()`
    (without strict) so custom overrides (like TestModel.load_state_dict) are used.

    Returns: (loaded_keys, skipped_keys)
    """
    if isinstance(checkpoint_state_dict, dict) and "state_dict" in checkpoint_state_dict:
        checkpoint_state_dict = checkpoint_state_dict["state_dict"]

    model_dict = model.state_dict()
    to_load = {}
    loaded = []
    skipped = []

    for ck_key, ck_val in checkpoint_state_dict.items():
        # try exact key
        if ck_key in model_dict and model_dict[ck_key].shape == ck_val.shape:
            to_load[ck_key] = ck_val
            loaded.append(ck_key)
            continue
        # try removing 'module.' prefix or adding it
        alt = None
        if ck_key.startswith('module.'):
            alt_key = ck_key[len('module.'):]
            if alt_key in model_dict and model_dict[alt_key].shape == ck_val.shape:
                alt = alt_key
        else:
            alt_key = 'module.' + ck_key
            if alt_key in model_dict and model_dict[alt_key].shape == ck_val.shape:
                alt = alt_key
        if alt is not None:
            to_load[alt] = ck_val
            loaded.append(f"{ck_key} -> {alt}")
        else:
            skipped.append(ck_key)

    # Update the model_dict with matched tensors and load
    if len(to_load) == 0:
        if verbose:
            print("No matching keys found between checkpoint and model. Nothing loaded.")
        return loaded, skipped

    # Create a new state dict to load: start from model's own state dict and update
    new_state = model_dict.copy()
    new_state.update(to_load)

    # Use model.load_state_dict so any override in model is used; non-strict so missing keys are allowed
    try:
        model.load_state_dict(new_state)
    except Exception as e:
        # As a fallback, attempt non-strict load directly
        try:
            model.load_state_dict(new_state, strict=False)
        except Exception:
            if verbose:
                print("Warning: failed to load state dict even in non-strict mode:", e)
            raise

    if verbose:
        print(f"Loaded {len(loaded)} keys from checkpoint; skipped {len(skipped)} keys.")
        if len(skipped) > 0:
            print("Example skipped keys:", skipped[:5])
    return loaded, skipped

def train_one_epoch(
    model, criterion, train_dataloader, optimizer, epoch, clip_max_norm, stage, noisequant=True,
):
    model.train()
    device = next(model.parameters()).device
    train_loss = AverageMeter()
    train_bpp_loss = AverageMeter()
    train_y_bpp_loss = AverageMeter()
    train_z_bpp_loss = AverageMeter()
    train_mse_loss = AverageMeter()
    start = time.time()
    for i, d in enumerate(train_dataloader):
        # print("d.shape:", d.shape) #[4,3,256,256]
        d = d.to(device)

        optimizer.zero_grad()
        out_net = model(d, noisequant)

        out_criterion = criterion(out_net, d)
        train_bpp_loss.update(out_criterion["bpp_loss"].item())
        train_y_bpp_loss.update(out_criterion["y_bpp_loss"].item())
        train_z_bpp_loss.update(out_criterion["z_bpp_loss"].item())
        train_loss.update(out_criterion["loss"].item())
        train_mse_loss.update(out_criterion["mse_loss"].item())
        if stage == 1:
            out_criterion["loss"].backward()
        else: 
            out_criterion["mse_loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        if i % 8 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.3f} |'
                f'\ty_Bpp loss: {out_criterion["y_bpp_loss"].item():.4f} |'
                f'\tz_Bpp loss: {out_criterion["z_bpp_loss"].item():.4f} |'
            )
    print(f"Train epoch {epoch}: Average losses:"
          f"\tLoss: {train_loss.avg:.3f} |"
          f"\tMSE loss: {train_mse_loss.avg:.3f} |"
          f"\tBpp loss: {train_bpp_loss.avg:.4f} |"
          f"\ty_Bpp loss: {train_y_bpp_loss.avg:.5f} |"
          f"\tz_Bpp loss: {train_z_bpp_loss.avg:.5f} |"
          f"\tTime (s) : {time.time()-start:.4f} |"
          )


    return train_loss.avg, train_bpp_loss.avg, train_mse_loss.avg

def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    y_bpp_loss = AverageMeter()
    z_bpp_loss = AverageMeter()
    mse_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            bpp_loss.update(out_criterion["bpp_loss"].item())
            y_bpp_loss.update(out_criterion["y_bpp_loss"].item())
            z_bpp_loss.update(out_criterion["z_bpp_loss"].item())
            loss.update(out_criterion["loss"].item())
            mse_loss.update(out_criterion["mse_loss"].item())

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.4f} |"
        f"\ty_Bpp loss: {y_bpp_loss.avg:.4f} |"
        f"\tz_Bpp loss: {z_bpp_loss.avg:.4f} |"
    )

    return loss.avg, bpp_loss.avg, mse_loss.avg

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def first_stage_lr_schedule_for_epoch(epoch: int, max_epochs: int):
    # LR stages list: {50,10,7.5,5,2.5,1} x 1e-5
    lr_values = [50e-5, 10e-5, 7.5e-5, 5e-5, 2.5e-5, 1e-5]
    stages = len(lr_values)
    seg = max(1, max_epochs // stages)
    idx = min(epoch // seg, stages - 1)
    return lr_values[idx]

# add 2 stage training
def two_stage_training(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, transform=train_transforms, mode="train",
                                split_ratio=args.split_ratio, filelist_json=args.split_json)
    test_dataset = ImageFolder(args.dataset, transform=test_transforms, mode="test",
                               split_ratio=args.split_ratio, filelist_json=args.split_json)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = TestModel(N=args.N, M=args.M, num_slices=5, flag_inference=False)
    net = net.to(device)
    # if not os.path.exists(args.savepath):
    #     try:
    #         os.mkdir(args.savepath)
    #     except:
    #         os.makedirs(args.savepath)
    try:
        # 使用 os.makedirs(exist_ok=True) 創建所有父目錄
        os.makedirs(args.savepath, exist_ok=True)
        print(f"儲存目錄成功創建/檢查: {args.savepath}")
    except Exception as e:
        print(f"FATAL ERROR: 無法創建儲存目錄 {args.savepath}. 請檢查權限或空間。錯誤訊息: {e}")
        # 如果無法創建目錄，則不能繼續寫入 SummaryWriter 或 Checkpoint
        return 
    writer = SummaryWriter(args.savepath)
    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    # if args.checkpoint:  # load from previous checkpoint
    #     print("Loading", args.checkpoint)
    #     checkpoint = torch.load(args.checkpoint, map_location=device)
    #     # print('checkpoint.keys:', checkpoint.keys())
    #     net.load_state_dict(checkpoint, strict=False)
    #     optimizer.load_state_dict(checkpoint["optimizer"])

    # Load checkpoint if specified (after optimizer creation)
    if args.checkpoint:
        print("Loading checkpoint:", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        # Load model weights using parent class method to support strict parameter first
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        # Get the underlying model if wrapped in DataParallel
        model_to_load = net.module if isinstance(net, (nn.DataParallel, CustomDataParallel)) else net
        try:
            # Try strict=True first
            nn.Module.load_state_dict(model_to_load, state_dict, strict=True)
            print("Checkpoint loaded with strict=True")
        except RuntimeError as e:
            print("Strict load failed:", e)
            print("Falling back to compatible loader (match by name+shape)...")
            loaded, skipped = load_pretrained_compat(model_to_load, state_dict, verbose=True)
            print(f"Compatibility loader loaded {len(loaded)} keys; skipped {len(skipped)} keys.")
        # Load optimizer if not pretrained-only
        if not args.pretrained and isinstance(checkpoint, dict) and "optimizer" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
                print("Optimizer state restored.")
            except Exception as e:
                print(f"Warning: Could not restore optimizer state: {e}")

    # if args.checkpoint and args.pretrained:
    #     # close scheduler
    #     del lr_scheduler
    # else:
    #     print("No pretrained model.")

    best_loss = float("inf")
    if args.stage == 1:
        print("Stage 1 training.")
        print(args)
        
        trainable_param_names = net.set_trainable_adapters(stage=1)
        # resolve names -> parameter tensors if needed
        trainable_params = resolve_trainable_params(net, trainable_param_names)
        print(f"Trainable parameters (count={len(trainable_params)}): {trainable_param_names if len(trainable_param_names)<20 else 'list(...)'}")
        # Use differentiable quantization approximation to imitate quantization
        # Use uniform noise addition during training
        noisequant = True 
        print("noisequant: {}".format(noisequant))

        # Create optimizer with initial lr from schedule
        init_lr = first_stage_lr_schedule_for_epoch(0, args.epochs)
        optimizer = torch.optim.Adam(trainable_params, lr=init_lr)
        criterion = RateDistortionLoss(lmbda=args.lmbda)
        
        for epoch in range(args.epochs):
            lr = first_stage_lr_schedule_for_epoch(epoch, args.epochs)
            optimizer.param_groups[0]['lr'] = lr

            print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
            train_loss, train_bpp, train_mse = train_one_epoch(
                net,
                criterion,
                train_dataloader,
                optimizer,
                epoch,
                args.clip_max_norm,
                args.stage,
                noisequant,
            )
            writer.add_scalar('Train/loss', train_loss, epoch)
            writer.add_scalar('Train/mse', train_mse, epoch)
            writer.add_scalar('Train/bpp', train_bpp, epoch)

            loss, bpp, mse = test_epoch(epoch, test_dataloader, net, criterion)
            writer.add_scalar('Test/loss', loss, epoch)
            writer.add_scalar('Test/mse', mse, epoch)
            writer.add_scalar('Test/bpp', bpp, epoch)

            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            if args.save:
                # 處理多 GPU 存檔 (必須在存檔前處理 state_dict)
                if isinstance(net, (torch.nn.DataParallel, CustomDataParallel)):
                    state_dict_to_save = net.module.state_dict()
                else:
                    state_dict_to_save = net.state_dict()
                # delete the file beggings with 'checkpoint_last'
                DelfileList(args.savepath, "checkpoint_last")
                save_checkpoint(
                    {
                        "epoch": epoch,
                        # "state_dict": net.state_dict(),
                        "state_dict": state_dict_to_save, 
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                    },
                    filename=os.path.join(args.savepath, "checkpoint_last_{}.pth.tar".format(epoch))
                )
                if is_best:
                    DelfileList(args.savepath, "checkpoint_best")
                    save_checkpoint(
                        {
                            "epoch": epoch,
                            # "state_dict": net.state_dict(),
                            "state_dict": state_dict_to_save, 
                            "loss": loss,
                            "optimizer": optimizer.state_dict(),
                        },
                        filename=os.path.join(args.savepath, "checkpoint_best_loss_{}.pth.tar".format(epoch))
                    )

    elif args.stage == 2:
        print("Stage 2 training.")
        print(args)
        trainable_param_names = net.set_trainable_adapters(stage=2)
        trainable_params = resolve_trainable_params(net, trainable_param_names)
        print(f"Trainable parameters (count={len(trainable_params)}): {trainable_param_names if len(trainable_param_names)<20 else 'list(...)'}")

        # Use hard quantization during training
        # Use STE (Straight-Through Estimator)
        noisequant = False 
        print("noisequant: {}".format(noisequant))

        # Create optimizer with initial lr from schedule
        init_lr = 5e-4
        print("learning rate: {}".format(init_lr))
        optimizer = torch.optim.Adam(trainable_params, lr=init_lr)
        criterion = RateDistortionLoss(lmbda=args.lmbda)
       
        
        for epoch in range(args.epochs):
            train_loss, train_bpp, train_mse = train_one_epoch(
                net,
                criterion,
                train_dataloader,
                optimizer,
                epoch,
                args.clip_max_norm,
                args.stage,
                noisequant,
            )
            writer.add_scalar('Train/loss', train_loss, epoch)
            writer.add_scalar('Train/mse', train_mse, epoch)
            writer.add_scalar('Train/bpp', train_bpp, epoch)

            loss, bpp, mse = test_epoch(epoch, test_dataloader, net, criterion)
            writer.add_scalar('Test/loss', loss, epoch)
            writer.add_scalar('Test/mse', mse, epoch)
            writer.add_scalar('Test/bpp', bpp, epoch)

            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            if args.save:
                # 處理多 GPU 存檔 (必須在存檔前處理 state_dict)
                if isinstance(net, (torch.nn.DataParallel, CustomDataParallel)):
                    state_dict_to_save = net.module.state_dict()
                else:
                    state_dict_to_save = net.state_dict()

                # delete the file beggings with 'checkpoint_last'
                DelfileList(args.savepath, "checkpoint_last")
                save_checkpoint(
                    {
                        "epoch": epoch,
                        # "state_dict": net.state_dict(),
                        "state_dict": state_dict_to_save,
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                    },
                    filename=os.path.join(args.savepath, "checkpoint_last_{}.pth.tar".format(epoch))
                )
                if is_best:
                    DelfileList(args.savepath, "checkpoint_best")
                    save_checkpoint(
                        {
                            "epoch": epoch,
                            # "state_dict": net.state_dict(),
                            "state_dict": state_dict_to_save,
                            "loss": loss,
                            "optimizer": optimizer.state_dict(),
                        },
                        filename=os.path.join(args.savepath, "checkpoint_best_loss_{}.pth.tar".format(epoch))
                    )

    else:
        raise ValueError("Invalid stage. Please choose stage 1 or 2.")

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "--N",
        default=192,
        type=int,
        help="Number of channels of main codec",
    )
    parser.add_argument(
        "--M",
        default=320,
        type=int,
        help="Number of channels of latent",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=750,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        required=True,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1, # original 32
        help="Test batch size (default: %(default)s)",
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", default=True, action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", default=1926, type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="use the pretrain model to refine the models",
    )
    parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--savepath', default='./checkpoint', type=str, help='Path to save the checkpoint')
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--stage", type=int, required=True, help="Training stage")
    parser.add_argument("--split-json", dest="split_json", type=str, default=None,
                        help="Path to a JSON file listing dataset filenames (overrides folder scan)")
    parser.add_argument("--split-ratio", dest="split_ratio", type=float, default=0.8,
                        help="Train split ratio when splitting dataset (default: 0.8 => 4:1)")
    args = parser.parse_args(argv)
    return args
    

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    two_stage_training(args)