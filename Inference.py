"""
Evaluate an end-to-end compression model on an image dataset.
"""
import argparse
import json
import sys
import time
import csv

from collections import defaultdict
from typing import List
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms
import torchvision
import compressai
from compressai.zoo import load_state_dict
import torch
import os
import math
import torch.nn as nn
from Network import TestModel
torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)

def collect_images(rootpath: str) -> List[str]:
    return [
        os.path.join(rootpath, f)
        for f in os.listdir(rootpath)
        if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS
    ]

def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.nn.functional.mse_loss(a, b).item()
    return -10 * math.log10(mse)

def read_image(filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")
    return transforms.ToTensor()(img)


@torch.no_grad()
def inference(model, x, f, outputpath, patch):
    x = x.unsqueeze(0)
    imgpath = f.split('/')
    imgPath = outputpath + '/rec/' + imgpath[-1]
    csvfile = outputpath + '/result.csv'
    print('decoding img: {}'.format(f))
########original padding
    h, w = x.size(2), x.size(3)
    p = patch  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = 0
    padding_right = new_w - w - padding_left
    padding_top = 0
    padding_bottom = new_h - h - padding_top
    pad = nn.ConstantPad2d((padding_left, padding_right, padding_top, padding_bottom), 0)
    x_padded = pad(x)

    _, _, height, width = x_padded.size()
    start = time.time()
    out_enc = model.compress(x_padded)
    enc_time = time.time() - start

    start = time.time()
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
    dec_time = time.time() - start

    out_dec["x_hat"] = torch.nn.functional.pad(
        out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = 0
    for s in out_enc["strings"]:
        for j in s:
            if isinstance(j, list):
                for i in j:
                    if isinstance(i, list):
                        for k in i:
                            bpp += len(k)
                    else:
                        bpp += len(i)
            else:
                bpp += len(j)
    bpp *= 8.0 / num_pixels
    # bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
    z_bpp = len(out_enc["strings"][1][0])* 8.0 / num_pixels
    y_bpp = bpp - z_bpp

    # Ensure output directory exists before saving reconstructed image
    os.makedirs(os.path.dirname(imgPath), exist_ok=True)
    torchvision.utils.save_image(out_dec["x_hat"], imgPath, nrow=1)
    PSNR = psnr(x, out_dec["x_hat"])
    # Ensure CSV directory exists (safety if called outside eval pipeline)
    os.makedirs(os.path.dirname(csvfile), exist_ok=True)
    with open(csvfile, 'a+') as f:
        row = [imgpath[-1], bpp * num_pixels, num_pixels, bpp, y_bpp, z_bpp,
               torch.nn.functional.mse_loss(x, out_dec["x_hat"]).item() * 255 ** 2, psnr(x, out_dec["x_hat"]),
               ms_ssim(x, out_dec["x_hat"], data_range=1.0).item(), enc_time, dec_time, out_enc["time"]['y_enc'] * 1000,
               out_dec["time"]['y_dec'] * 1000, out_enc["time"]['z_enc'] * 1000, out_enc["time"]['z_dec'] * 1000,
               out_enc["time"]['params'] * 1000]
        write = csv.writer(f)
        write.writerow(row)
    print('bpp:{}, PSNR: {}, encoding time: {}, decoding time: {}'.format(bpp, PSNR, enc_time, dec_time))
    return {
        "psnr": PSNR,
        "bpp": bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }

@torch.no_grad()
def inference_entropy_estimation(model, x, f, outputpath, patch):
    x = x.unsqueeze(0)
    imgpath = f.split('/')
    imgPath = outputpath + '/rec/' + imgpath[-1]
    csvfile = outputpath + '/result.csv'
    print('decoding img: {}'.format(f))
    ########original padding
    h, w = x.size(2), x.size(3)
    p = patch  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = torch.nn.functional.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    _, _, height, width = x_padded.size()


    start = time.time()
    out_net = model.inference(x_padded)


    elapsed_time = time.time() - start
    out_net["x_hat"] = torch.nn.functional.pad(
        out_net["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out_net["likelihoods"].values()
    )
    y_bpp = (torch.log(out_net["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels))
    z_bpp = (torch.log(out_net["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels))

    # Ensure output directory exists before saving reconstructed image
    os.makedirs(os.path.dirname(imgPath), exist_ok=True)
    torchvision.utils.save_image(out_net["x_hat"], imgPath, nrow=1)
    PSNR = psnr(x, out_net["x_hat"])
    # Ensure CSV directory exists (safety if called outside eval pipeline)
    os.makedirs(os.path.dirname(csvfile), exist_ok=True)
    with open(csvfile, 'a+') as f:
        row = [imgpath[-1], bpp.item() * num_pixels, num_pixels, bpp.item(), y_bpp.item(), z_bpp.item(),
               torch.nn.functional.mse_loss(x, out_net["x_hat"]).item() * 255 ** 2, PSNR,
               ms_ssim(x, out_net["x_hat"], data_range=1.0).item(), elapsed_time / 2.0, elapsed_time / 2.0,
               out_net["time"]['y_enc'] * 1000, out_net["time"]['y_dec'] * 1000, out_net["time"]['z_enc'] * 1000,
               out_net["time"]['z_dec'] * 1000, out_net["time"]['params'] * 1000]
        write = csv.writer(f)
        write.writerow(row)
    return {
        "psnr": PSNR,
        "bpp": bpp.item(),
        "encoding_time": elapsed_time / 2.0,  # broad estimation
        "decoding_time": elapsed_time / 2.0,
    }


def eval_model(model, filepaths, entropy_estimation=False, half=False, outputpath='Recon', patch=576):
    device = next(model.parameters()).device
    metrics = defaultdict(float)
    imgDir = outputpath
    if not os.path.isdir(imgDir):
        os.makedirs(imgDir)
    csvfile = imgDir + '/result.csv'
    if os.path.isfile(csvfile):
        os.remove(csvfile)
    with open(csvfile, 'w') as f:
        row = ['name', 'bits', 'pixels', 'bpp', 'y_bpp', 'z_bpp', 'mse', 'psnr(dB)', 'ms-ssim', 'enc_time(s)', 'dec_time(s)', 'y_enc(ms)',
               'y_dec(ms)', 'z_enc(ms)', 'z_dec(ms)', 'param(ms)']
        write = csv.writer(f)
        write.writerow(row)
    for f in filepaths:
        x = read_image(f).to(device)
        if not entropy_estimation:
            if half:
                model = model.half()
                x = x.half()
            rv = inference(model, x, f, outputpath, patch)
        else:
            rv = inference_entropy_estimation(model, x, f, outputpath, patch)
        for k, v in rv.items():
            metrics[k] += v
    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)
    return metrics

def setup_args():
    parser = argparse.ArgumentParser(
        add_help=False,
    )

    # Common options.
    parser.add_argument("--dataset", type=str, help="dataset path")
    parser.add_argument(
        "--output_path",
        help="result output path",
    )
    parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="enable CUDA",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="convert model to half floating point (fp16)",
    )
    parser.add_argument(
        "--entropy-estimation",
        action="store_true",
        help="use evaluated entropy estimation (no entropy coding)",
    )
    parser.add_argument(
        "-p",
        "--path",
        dest="paths",
        type=str,
        required=True,
        help="checkpoint path",
    )
    parser.add_argument(
        "--patch",
        type=int,
        default=256,
        help="padding patch size (default: %(default)s)",
    )
    parser.add_argument(
        "--inference-json",
        dest="inference_json",
        type=str,
        default=None,
        help="Path to a JSON file listing filenames to use for inference (overrides directory scan)",
    )
    return parser


def main(argv):
    parser = setup_args()
    args = parser.parse_args(argv)
# ===================================== added
    # If an inference JSON is provided, try to load filenames from it (supports several common layouts)
    filepaths = []
    if getattr(args, 'inference_json', None):
        try:
            with open(args.inference_json, 'r', encoding='utf-8') as f:
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

            # normalize and join with dataset root if needed
            for fn in filenames:
                if os.path.isabs(fn):
                    candidate = fn
                else:
                    candidate = os.path.join(args.dataset, fn)
                if os.path.exists(candidate):
                    filepaths.append(candidate)
                else:
                    # try basename only search in dataset
                    alt = os.path.join(args.dataset, os.path.basename(fn))
                    if os.path.exists(alt):
                        filepaths.append(alt)
                    else:
                        print(f"Warning: listed file not found: {fn}")

            filepaths = sorted(filepaths)
        except Exception as e:
            print(f"Failed to load inference JSON '{args.inference_json}': {e}")
            filepaths = collect_images(args.dataset)
            filepaths = sorted(filepaths)
    else:
        filepaths = collect_images(args.dataset)
        filepaths = sorted(filepaths)

    if len(filepaths) == 0:
        print("Error: no images found for inference.", file=sys.stderr)
        sys.exit(1)
# =====================================
    compressai.set_entropy_coder(args.entropy_coder)

    # Load checkpoint and normalize its state_dict to avoid KeyError when
    # updating registered buffers (e.g. gaussian_conditional._quantized_cdf).
    # Support common forms: plain state_dict, dict containing 'state_dict',
    # and keys with 'module.' prefix.
    raw_ckpt = torch.load(args.paths, map_location="cpu")
    if isinstance(raw_ckpt, dict) and "state_dict" in raw_ckpt:
        sd = raw_ckpt["state_dict"]
    else:
        sd = raw_ckpt

    # If load_state_dict helper returned an object (from compressai.zoo),
    # use it directly; otherwise ensure we have a mapping of param names -> tensors.
    if hasattr(sd, "items"):
        normalized = {}
        for k, v in sd.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            normalized[nk] = v
        state_dict = normalized
    else:
        state_dict = sd

    model_cls = TestModel(flag_inference=True)
    model = model_cls.from_state_dict(state_dict).eval()

    results = defaultdict(list)

    if args.cuda and torch.cuda.is_available():
        model = model.to("cuda")

    # Ensure entropy models' internal CDFs / buffers are initialized.
    # CompressAI may require calling `update()` after loading a checkpoint
    # to initialize quantized CDFs. If these are uninitialized you'll see
    # "Uninitialized CDFs. Run update() first" errors during compress().
    try:
        # model.update returns True if any buffers were updated
        updated = model.update()
        print(f"model.update() -> {updated}")
    except Exception as e:
        print("Warning: model.update() failed:", e)

    metrics = eval_model(model, filepaths, args.entropy_estimation, args.half, args.output_path, args.patch)
    for k, v in metrics.items():
        results[k].append(v)

    description = (
        "entropy estimation" if args.entropy_estimation else args.entropy_coder
    )
    output = {
        "description": f"Inference ({description})",
        "results": results,
    }
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main(sys.argv[1:])



