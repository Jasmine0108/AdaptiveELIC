import argparse
import pathlib
from PIL import Image

def resize_and_center_crop(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """Resize the image so both dimensions are >= target, then center-crop.

    This uses a scale that ensures the resized image fully covers the target
    rectangle, then crops the center to (target_width, target_height).
    """
    width, height = image.size
    # scale so both dimensions are at least the target dimensions
    scale = max(target_width / width, target_height / height)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    resized = image.resize((new_width, new_height), resample=Image.Resampling.BICUBIC)
    left = (new_width - target_width) // 2
    upper = (new_height - target_height) // 2
    right = left + target_width
    lower = upper + target_height
    return resized.crop((left, upper, right, lower))


def process_directory(input_dir: pathlib.Path, output_dir: pathlib.Path, target_width: int, target_height: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    supported_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in supported_ext:
            with Image.open(path) as img:
                processed = resize_and_center_crop(img.convert("RGB"), target_width, target_height)
                processed.save(output_dir / path.name)
        elif path.is_dir():
            process_directory(path, output_dir / path.name, target_width, target_height)

def main() -> None:
    parser = argparse.ArgumentParser(description="Resize and center-crop images to target width and height.")
    parser.add_argument("--input-dir", type=pathlib.Path, required=True, help="輸入影像資料夾")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True, help="輸出影像資料夾")
    parser.add_argument("--width", type=int, required=True, help="Input target width")
    parser.add_argument("--height", type=int, required=True, help="Input target height")
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("Width and height must be positive integers.")
    process_directory(args.input_dir, args.output_dir, args.width, args.height)

if __name__ == "__main__":
    main()