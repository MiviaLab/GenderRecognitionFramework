import os
import sys
from glob import glob
from tqdm import tqdm
from cv2 import cv2
import numpy as np
from multiprocessing import Pool

sys.path.append("../training")
from ferplus_aug_dataset import MyCustomAugmentation, corruptions

EXT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "data"

def show_one_image(dirin="gender-access/lfw_cropped"):
    dirin = os.path.join(os.path.join(EXT_ROOT, DATA_DIR), dirin)
    impath = glob(os.path.join(dirin, '*'))[0]
    imex = cv2.imread(impath)
    TARGET_SHAPE = (256, 256, 3)
    P = 'test'
    print('Partition: %s' % P)
    while True:
        NUM_LEVELS = 5
        imout = np.zeros((TARGET_SHAPE[1] * NUM_LEVELS, TARGET_SHAPE[0] * len(corruptions), 3), dtype=np.uint8)
        print(imout.shape)
        for ind1, ctypes in enumerate(corruptions):
            for ind2 in range(NUM_LEVELS):
                a = MyCustomAugmentation(ctypes, [ind2] * len(ctypes))
                imex_corrupted = a.before_cut(imex, None)
                off1 = ind1 * TARGET_SHAPE[0]
                off2 = ind2 * TARGET_SHAPE[1]
                imout[off2:off2 + TARGET_SHAPE[1], off1:off1 + TARGET_SHAPE[0], :] = imex_corrupted

        # imout = cv2.resize(imout, (TARGET_SHAPE[0]*2, TARGET_SHAPE[1]*2))
        cv2.imshow('imout', imout)
        k = cv2.waitKey(0)
        if k == 27:
            sys.exit(0)


def export_dataset(augmentation,
                   dirout='corrupted_lfw_plus_dataset/lfw_plus.%s.%s/',
                   dirin='gender-access/lfw_cropped',
                   partition='all'):

    dirin = os.path.join(os.path.join(EXT_ROOT, DATA_DIR), dirin)
    dirout = os.path.join(os.path.join(EXT_ROOT, DATA_DIR), dirout)

    print("From", dirin)
    print("To", dirout)
    
    dirout = dirout % (partition, str(augmentation))
    if not os.path.exists(dirout): os.mkdir(dirout)
    images = [x for x in glob(os.path.join(dirin, '*')) if partition is "all" or os.path.basename(x).startswith(partition + '_')]
    for inim in tqdm(images):
        im = cv2.imread(inim)
        if im is not None:
            imo = augmentation.before_cut(im, (0, 0, im.shape[0], im.shape[1]))
            outim = os.path.join(dirout, inim[len(dirin) + 1:])
            cv2.imwrite(outim, imo)
    return dirout


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


if '__main__' == __name__:
    if len(sys.argv) > 1 and sys.argv[1].startswith('exp'):
        print("Exporting dataset")
        export_datasets()
    else:
        show_one_image()
