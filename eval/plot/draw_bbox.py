
import os
import argparse

from pathlib import Path
from glob import glob
from PIL import Image, ImageDraw, ImageOps

def draw_bbox(image_path, out_path, color, width, x1, y1, x2, y2):
    """Draw bounding box"""
    im = Image.open(image_path)
    draw = ImageDraw.Draw(im)
    draw.rectangle([(x1, y1), (x2, y2)], fill=None, outline=color, width=width)
    im.save(out_path, 'png')

def draw_border(image_path, out_path, width, color):
    img = Image.open(image_path)
    img_with_border = ImageOps.expand(img, border=width, fill=color)
    img_with_border.save(out_path)

def crop_image(image_path, out_path, x1, y1, x2, y2):
    """Crop image with specific range"""
    img = Image.open(image_path)
    cropped_img = img.crop((x1, y1, x2, y2))
    cropped_img.save(out_path)

def crop_bound_image(image_path, out_path, top, bottom, left, right):
    """Crop image with top, bottom, left, right bounding"""
    img = Image.open(image_path)
    width, height = img.width, img.height
    # area = (left, top, width - right, height - bottom)
    # cropped_img = img.crop(area)
    # cropped_img.save(out_path)
    area1 = (left, top, width - right, (height - top - bottom) // 2 + top)
    cropped_img1 = img.crop(area1)
    cropped_img1.save(str(out_path).replace(".png", "-i.png"))

    area2 = (left, (height - top - bottom) // 2 + top, width - right, height - bottom)
    cropped_img2 = img.crop(area2)
    cropped_img2.save(str(out_path).replace(".png", "-o.png"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--color', type=str, required=True)
    parser.add_argument('--bbox_width', type=int, default=3)
    parser.add_argument('--left', type=int, required=True)
    parser.add_argument('--top', type=int, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    args = parser.parse_args()

    color = '#6699a3' # '#336699' # '#6699a3'

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    input_paths = glob(f'{args.input_dir}/*.png')

    for input_path in input_paths:
        _, filename = os.path.split(input_path)
        output_path = Path(args.output_dir) / filename
        draw_bbox(input_path, output_path, args.color, args.bbox_width, args.left, args.top, args.left + args.width, args.top + args.height)
