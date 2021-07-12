import os
import argparse
from shutil import copyfile
import numpy as np

def rename(path):
    folder_path = path
    source_path = path + "/bell_papper/rgb"

    for img in os.listdir(source_path):
        if img.startswith("bell_papper"):
            split = img.split("bell_papper")
            new_img = "bell_pepper" + split[1]
            os.rename(os.path.join(source_path, img), os.path.join(source_path, new_img))
            print(f'{img} -> {new_img}\n')

    os.rename(os.path.join(folder_path, "bell_papper"), os.path.join(folder_path, "bell_pepper"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to transform ROD into Image Folder')
    parser.add_argument('--path', type=str, default="../data/synARID_crops_square/synARID_crops_square",
                        help="The input folder")

    args = parser.parse_args()
    path = args.path
    rename(path)
