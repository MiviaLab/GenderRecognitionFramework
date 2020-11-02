#!/usr/bin/python3
import os
import sys
import pickle
from glob import glob
from tqdm import tqdm
from cv2 import cv2
import numpy as np
import random
import json
from multiprocessing import Pool

sys.path.append("../training")
from ferplus_aug_dataset import MyCustomAugmentation, corruptions

from face_detector import enclosing_square, add_margin, cut
from feret_dataset_gender import FERETDatasetGender as Dataset, FEMALE_LABEL, MALE_LABEL

'''
The script provides for an already cropped version of FERET AUGMENTED DATASET
'''


EXT_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = "cache"
DATA_DIR = "data"


def load_rois(path):
    with open(path) as f:
        return json.load(f)


def mkdir_recursive(tree):
    os.makedirs(tree, exist_ok=True)


def export_dataset(augmentation,
                   dirout='corrupted_feret_gender_dataset/feret_augmentation.{augmentation}/{gender}/{partition}_set',
                   dirin='gender-feret/{gender}/{partition}_set',
                   partition='test'):

    assert partition in ["train", "test"], "Unrecognized partition {}".format(partition)
    print(str(augmentation))

    dirin = os.path.join(os.path.join(EXT_ROOT, DATA_DIR), dirin)

    female_dirout = dirout.format(partition=partition, augmentation=str(augmentation), gender="female")
    male_dirout = dirout.format(partition=partition, augmentation=str(augmentation), gender="male")

    female_dirout = os.path.join(os.path.join(EXT_ROOT, DATA_DIR), female_dirout)
    male_dirout = os.path.join(os.path.join(EXT_ROOT, DATA_DIR), male_dirout)

    mkdir_recursive(female_dirout)
    mkdir_recursive(male_dirout)

    if not os.path.exists(female_dirout): os.mkdir(female_dirout)
    if not os.path.exists(male_dirout): os.mkdir(male_dirout)

    dv = Dataset(partition, augment=False)
    gen = dv.get_generator(fullinfo=True)

    for batch in tqdm(gen):
        for _, label, path, roi in zip(batch[0], batch[1], batch[2], batch[3]):     

            # DEBUG
            # imo = augmentation.before_cut(img, (0, 0, img.shape[0], img.shape[1]))
            # cv2.imshow("raw",cut(cv2.imread(path), roi))
            # cv2.imshow("raw corr", augmentation.before_cut(cv2.imread(path), roi))
            # cv2.imshow("img", img)
            # cv2.imshow("imo", imo)
            # if cv2.waitKey(0) & 0xFF == ord('q'):
            #         cv2.destroyAllWindows()

            img_raw = cut(cv2.imread(path), roi)
            imo = augmentation.before_cut(img_raw, None)

            name = os.path.split(path)[-1]
            directory = female_dirout if np.argmax(label) == FEMALE_LABEL else male_dirout
            cv2.imwrite(os.path.join(directory, name), imo)
    female_dirin = dirin.format(gender="female", partition=partition)
    male_dirin = dirin.format(gender="male", partition=partition)
    original_females = len(glob(os.path.join(female_dirin, '*')))
    original_males = len(glob(os.path.join(male_dirin, '*')))
    corrupted_females = len(glob(os.path.join(female_dirout, '*')))
    corrupted_males = len(glob(os.path.join(male_dirout, '*')))

    success = True

    if not corrupted_females == original_females:
        print("Error, not all female samples have been corrupted")
        print("Originals", original_females)
        print("Corrupted", corrupted_females)
        success = False

    if not corrupted_males == original_males:
        print("Error, not all male samples have been corrupted")
        print("Originals", original_males)
        print("Corrupted", corrupted_males)
        success = False

    assert success, "Incomplete corruption {}".format(dirout)

    return str(augmentation)


def export_datasets():
    NUM_LEVELS = 5
    aug_arr = []
    for corruption_types in corruptions:
        print(corruption_types)
        for corruption_qty in range(NUM_LEVELS):
            a = MyCustomAugmentation(corruption_types, [1 + corruption_qty] * len(corruption_types))
            aug_arr.append(a)
    p = Pool(5)
    p.map(export_dataset, aug_arr)


if '__main__' == __name__ and len(sys.argv) > 1 and sys.argv[1].startswith('exp'):
    print("Exporting dataset")
    export_datasets()
