
import os
import argparse
import math

from pathlib import Path
from glob import glob
from PIL import Image

def transparent2white(in_path, out_path):

    img = Image.open(in_path)
    pixel_data = img.load()

    # if img.mode == "RGBA":
    #     # If the image has an alpha channel, convert it to white
    #     # Otherwise we'll get weird pixels
    #     for y in range(img.size[1]): # For each row ...
    #         for x in range(img.size[0]): # Iterate through each column ...
    #             # Check if it's opaque
    #             if pixel_data[x, y][3] < 255:
    #                 # Replace the pixel data with the colour white
    #                 pixel_data[x, y][3] = 255
    # else:
    #     assert False, "Image is not RGBA format"

    for y in range(img.size[1]):
        for x in range(img.size[0]):
            # print(type(pixel_data[x, y]))
            alpha = pixel_data[x, y][3] / 255.0
            r, g, b = pixel_data[x, y][0], pixel_data[x, y][1], pixel_data[x, y][2]
            # print(type(pixel_data[x, y][0]))
            # print(math.floor(r * alpha + 255.0 * (1.0 - alpha)))
            nr = math.floor(r * alpha + 255.0 * (1.0 - alpha))
            ng = math.floor(g * alpha + 255.0 * (1.0 - alpha))
            nb = math.floor(b * alpha + 255.0 * (1.0 - alpha))
            pixel_data[x, y] = (nr, ng, nb, 255)

    img.save(out_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    args = parser.parse_args()

    input_paths = glob(f'{args.input_dir}/*.png')

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    for input_path in input_paths:
        print(input_path)
        _, file_name = os.path.split(input_path)
        output_path = Path(args.output_dir) / file_name

        transparent2white(input_path, output_path)
