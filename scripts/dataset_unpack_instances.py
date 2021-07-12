import os
import argparse
from shutil import copyfile
import numpy as np


def unpack_instances(path, dataset='rgbd-dataset'):
    source_path = os.path.join(path, dataset, dataset)

    dest_path = os.path.join(path, dataset, f"{dataset}_reorganized")
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)

    for classname in os.listdir(source_path):
        if not "testinstance_ids.txt" == classname and not "train" == classname and not "val" == classname:
            print(classname)

            src_dir = source_path+"/"+classname

            for instdir in os.listdir(src_dir):
                dest = dest_path + "/" + classname
                src = src_dir + "/" + instdir
                print("\t " + src)

                if not os.path.exists(dest):
                    os.mkdir(dest)

                for image in os.listdir(src):
                    if "crop" in image and "mask" not in image and "depth" not in image:
                        copyfile(src+"/"+image, dest+"/"+image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to unpack instances for ARID and rgbd-dataset')
    parser.add_argument('--path', type=str, default="../data/", help="The input folder")
    parser.add_argument('--dataset', type=str, default="rgbd-dataset", choices=["rgbd-dataset", "arid_40k_dataset_crops"],
                        help="Name of the folder")

    args = parser.parse_args()
    unpack_instances(args.path, args.dataset)
