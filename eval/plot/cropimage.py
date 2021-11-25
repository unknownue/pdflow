
import os
import argparse

from pathlib import Path
from glob import glob
from PIL import Image

def crop_image(in_path, out_path, x1, y1, x2, y2):
    """Crop image with specific range"""

    img = Image.open(in_path)

    # print(x1, y1, x2, y2)
    cropped_img = img.crop((x1, y1, x2, y2))
    cropped_img.save(out_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--postfix', type=str, default=None)
    parser.add_argument('--left', type=int, required=True)
    parser.add_argument('--top', type=int, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    args = parser.parse_args()

    input_paths = glob(f'{args.input_dir}/*.png')

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    for input_path in input_paths:
        print(input_path)
        _, file_name = os.path.split(input_path)

        if args.postfix is None:
            output_path = Path(args.output_dir) / file_name
        else:
            output_path = Path(args.output_dir) / file_name.replace('.png', f'-{args.postfix}.png')

        crop_image(input_path, output_path, args.left, args.top, args.left + args.width, args.top + args.height)
