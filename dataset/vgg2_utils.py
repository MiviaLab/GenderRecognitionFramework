import numpy as np
import time
import random
import cv2
import sys
import os
import keras
from tqdm import tqdm

sys.path.append("../training")
from dataset_tools import _readcsv

NUM_CLASSES = 8631 + 500

PARTITION_TRAIN = 0
PARTITION_VAL = 1
PARTITION_TEST = 2

vgg2ids = None
ids2vgg = None


def _load_identities(idmetacsv):
    global vgg2ids
    global ids2vgg
    if ids2vgg is None:
        vgg2ids = {}
        ids2vgg = []
        arr = _readcsv(idmetacsv)
        i = 0
        for line in arr:
            try:
                vggnum = int(line[0][1:])
                vgg2ids[vggnum] = (line[1], i)
                ids2vgg.append((line[1], vggnum))
                i += 1
            except Exception:
                pass
        print(len(ids2vgg), len(vgg2ids), NUM_CLASSES)
        assert (len(ids2vgg) == NUM_CLASSES)
        assert (len(vgg2ids) == NUM_CLASSES)


def get_id_from_vgg2(vggidn, idmetacsv='vggface2/identity_meta.csv'):
    _load_identities(idmetacsv)
    try:
        return vgg2ids[vggidn]
    except KeyError:
        print('ERROR: n%d unknown' % vggidn)
        return 'unknown', -1


def get_vgg2_identity(idn, idmetacsv='vggface2/identity_meta.csv'):
    _load_identities(idmetacsv)
    try:
        return ids2vgg[idn]
    except IndexError:
        print('ERROR: %d unknown', idn)
        return 'unknown', -1







