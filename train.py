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

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
# from torchvision.datasets import ImageFolder

from tensorboardX import SummaryWriter
from PIL import ImageFile
from PIL import Image  # 這次報錯是因為缺了這一行
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True
from ELICUtilis.utilis.utilis import DelfileList, load_checkpoint
from Network import TestModel
from torch.utils.data import Dataset

import os 
import sys
import datetime # 確保 datetime 已經在最上面 import

class ImageFolder(Dataset):
    # 1. 增加 mode 參數和 split_ratio 參數 (預設 80% 訓練集)
    def __init__(self, root, transform=None, mode='train', split_ratio=0.8):
        self.root = root
        self.transform = transform
        self.mode = mode.lower() # 轉小寫確保判斷正確
        self.split_ratio = split_ratio
        
        # 定義支援的圖片格式
        valid_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        
        if not os.path.exists(root):
            raise FileNotFoundError(f"Root directory not found: {root}")
            
        # 讀取該資料夾下所有符合格式的檔案
        all_samples = [
            os.path.join(root, f) for f in os.listdir(root) 
            if f.lower().endswith(valid_extensions)
        ]
        
        # 排序確保 train/test 劃分在不同運行中保持一致
        all_samples.sort()

        if len(all_samples) == 0:
            raise RuntimeError(f"Found 0 files in {root}. Supported extensions are: {valid_extensions}")

        # 2. 執行 train/test 劃分邏輯
        total_count = len(all_samples)
        # 計算訓練集的數量 (取整數)
        train_count = math.floor(total_count * self.split_ratio) 
        
        if self.mode == 'train':
            # 訓練集：從開始到 train_count
            self.samples = all_samples[:train_count]
            print(f"Dataset initialized in TRAIN mode. Using {len(self.samples)} files.")
        elif self.mode == 'test':
            # 測試集：從 train_count 到結束
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


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate, betas=(0.9, 0.999),
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate, betas=(0.9, 0.999),
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, noisequant=True,
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
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        out_net = model(d, noisequant)

        out_criterion = criterion(out_net, d)
        train_bpp_loss.update(out_criterion["bpp_loss"].item())
        train_y_bpp_loss.update(out_criterion["y_bpp_loss"].item())
        train_z_bpp_loss.update(out_criterion["z_bpp_loss"].item())
        train_loss.update(out_criterion["loss"].item())
        train_mse_loss.update(out_criterion["mse_loss"].item())

        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 10000 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.3f} |'
                f'\ty_Bpp loss: {out_criterion["y_bpp_loss"].item():.4f} |'
                f'\tz_Bpp loss: {out_criterion["z_bpp_loss"].item():.4f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
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
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss().item())
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
        f"\tAux loss: {aux_loss.avg:.4f}\n"
    )

    return loss.avg, bpp_loss.avg, mse_loss.avg


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

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
        default=4000,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
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
        default=15e-3,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=32,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
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
    parser.add_argument('--gpu-id', default=None, type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--savepath', default=None, type=str, help='Path to save the checkpoint')
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    # ==========================================================
    # ★ 關鍵修正 1: 處理 GPU 環境變數，確保多卡能工作
    # ==========================================================
    # 只有當使用者顯式指定 gpu-id 時才設定環境變數，否則讓系統自動分配所有 GPU
    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    # 原始程式碼的 os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id 應該被刪除或替換為上面這段
    # ==========================================================
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False

    # ==========================================================
    # ★ 關鍵修正 2: 動態生成 savepath
    # ==========================================================
    if args.savepath is None:
        WORK_ROOT_DIR = "/work/yang920428/IG_final_logs"  
        
        # 取得資料夾名稱 (確保 os.path.basename 可以正常運作)
        dataset_name = os.path.basename(args.dataset.strip('/'))
        
        # 取得時間戳記
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 組合路徑
        args.savepath = os.path.join(
            WORK_ROOT_DIR, 
            dataset_name,
            time_str
        )
        print(f"動態設定儲存路徑為: {args.savepath}")
    # train_transforms = transforms.Compose(
    #     [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    # )

    # test_transforms = transforms.Compose(
    #     [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    # )

    train_transforms = transforms.Compose(
        [
            # 1. 將圖片縮放為 512x512
            transforms.Resize((512, 512)), 
            
            # 2. 隨機裁切出 args.patch_size (訓練時用，具備資料增強效果)
            transforms.RandomCrop(args.patch_size), 
            
            # 3. 轉為 Tensor
            transforms.ToTensor()
        ]
    )

    test_transforms = transforms.Compose(
        [
            # 1. 將圖片縮放為 512x512
            transforms.Resize((512, 512)),
            
            # 2. 從中心裁切出 args.patch_size (測試時用，保證一致性)
            transforms.CenterCrop(args.patch_size), 
            
            # 3. 轉為 Tensor
            transforms.ToTensor()
        ]
    )

    train_dataset = ImageFolder(args.dataset, transform=train_transforms, mode = "train")
    test_dataset = ImageFolder(args.dataset, transform=test_transforms, mode = "test")

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

    net = TestModel(N=args.N, M=args.M)
    net = net.to(device)
    # ==========================================================
    # ★ 關鍵修正 3: 處理目錄創建 (放在 SummaryWriter 之前)
    # ==========================================================
    # 刪除原本的 if not os.path.exists(args.savepath): try/except 區塊！
    
    try:
        # 使用 os.makedirs(exist_ok=True) 創建所有父目錄
        os.makedirs(args.savepath, exist_ok=True)
        print(f"儲存目錄成功創建/檢查: {args.savepath}")
    except Exception as e:
        print(f"FATAL ERROR: 無法創建儲存目錄 {args.savepath}. 請檢查權限或空間。錯誤訊息: {e}")
        # 如果無法創建目錄，則不能繼續寫入 SummaryWriter 或 Checkpoint
        return 

    writer = SummaryWriter(args.savepath) # 這裡現在保證目錄是存在的
    # if not os.path.exists(args.savepath):
    #     try:
    #         os.mkdir(args.savepath)
    #     except:
    #         os.makedirs(args.savepath)
    # writer = SummaryWriter(args.savepath)
    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=8)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3800], gamma=0.1)
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    stemode = False ##set the pretrained flag
    if args.checkpoint and args.pretrained:
        optimizer.param_groups[0]['lr'] = args.learning_rate
        aux_optimizer.param_groups[0]['lr'] = args.aux_learning_rate
        del lr_scheduler
        # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=10)
        last_epoch = 0
        stemode = True

    noisequant = True
    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        if epoch > 3800 or stemode:
            noisequant = False
        print("noisequant: {}, stemode:{}".format(noisequant, stemode))
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_loss, train_bpp, train_mse = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            noisequant
        )
        writer.add_scalar('Train/loss', train_loss, epoch)
        writer.add_scalar('Train/mse', train_mse, epoch)
        writer.add_scalar('Train/bpp', train_bpp, epoch)

        loss, bpp, mse = test_epoch(epoch, test_dataloader, net, criterion)
        writer.add_scalar('Test/loss', loss, epoch)
        writer.add_scalar('Test/mse', mse, epoch)
        writer.add_scalar('Test/bpp', bpp, epoch)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            # ★ 關鍵修正 4: 處理多 GPU 存檔 (必須在存檔前處理 state_dict)
            if isinstance(net, (torch.nn.DataParallel, CustomDataParallel)):
                state_dict_to_save = net.module.state_dict()
            else:
                state_dict_to_save = net.state_dict()
                
            DelfileList(args.savepath, "checkpoint_last")
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": state_dict_to_save, # 使用處理後的 state_dict
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                filename=os.path.join(args.savepath, "checkpoint_last_{}.pth.tar".format(epoch))
            )
            if is_best:
                DelfileList(args.savepath, "checkpoint_best")
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": state_dict_to_save, # 使用處理後的 state_dict
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    filename=os.path.join(args.savepath, "checkpoint_best_loss_{}.pth.tar".format(epoch))
                )
            # DelfileList(args.savepath, "checkpoint_last")
            # save_checkpoint(
            #     {
            #         "epoch": epoch,
            #         "state_dict": net.state_dict(),
            #         "loss": loss,
            #         "optimizer": optimizer.state_dict(),
            #         "aux_optimizer": aux_optimizer.state_dict(),
            #         "lr_scheduler": lr_scheduler.state_dict(),
            #     },
            #     filename=os.path.join(args.savepath, "checkpoint_last_{}.pth.tar".format(epoch))
            # )
            # if is_best:
            #     DelfileList(args.savepath, "checkpoint_best")
            #     save_checkpoint(
            #         {
            #             "epoch": epoch,
            #             "state_dict": net.state_dict(),
            #             "loss": loss,
            #             "optimizer": optimizer.state_dict(),
            #             "aux_optimizer": aux_optimizer.state_dict(),
            #             "lr_scheduler": lr_scheduler.state_dict(),
            #         },
            #         filename=os.path.join(args.savepath, "checkpoint_best_loss_{}.pth.tar".format(epoch))
            #     )

if __name__ == "__main__":
    main(sys.argv[1:])
