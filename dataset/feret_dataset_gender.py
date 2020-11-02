from cv2 import cv2
from tqdm import tqdm
import os
import pickle
import numpy as np
import csv
import sys
from glob import glob
from face_detector import FaceDetector, findRelevantFace #, enclosing_square, add_margin

sys.path.append("../training")
from dataset_tools import enclosing_square, add_margin, DataGenerator

EXT_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = "cache"
DATA_DIR = "data"

PARTITION_TEST = 2
NUM_CLASSES = 2

FEMALE_LABEL = 0
MALE_LABEL = 1

FACE_DETECTOR = None


def get_gender_label(gender_string):
    if gender_string.startswith("m"):
        return MALE_LABEL
    elif gender_string.startswith("f"):
        return FEMALE_LABEL
    else:
        return None


def get_gender_string(label):
    if label == MALE_LABEL:
        return "male"
    elif label == FEMALE_LABEL:
        return "female"
    else:
        return label


def detect_face_caffe(frame):
    global FACE_DETECTOR 
    if FACE_DETECTOR is None:
        FACE_DETECTOR = FaceDetector(min_confidence=0.7)
    face = findRelevantFace(FACE_DETECTOR.detect(frame), frame.shape[1], frame.shape[0])
    if face is None:
        return None
    roi = enclosing_square(face['roi'])
    roi = add_margin(roi, 0.2)
    return roi

def entire_roi(img):
    return [0, 0, img.shape[1], img.shape[0]]

def _load_feret(imagesdir, partition="test", debug_max_num_samples=None, detect_face=True):
    data = list()
    discarded = {"male":0, "female":0}
    if partition.startswith("train") or partition.startswith("val"):
        dir_partition = "training"
    else:
        dir_partition = partition
    imagesdir = imagesdir.replace('<part>', dir_partition)
    for gender in ["male", "female"]:
        gender_image_dir = imagesdir.replace('<gender>', gender)
        category_label = get_gender_label(gender)
        for n, path in enumerate(tqdm(glob(os.path.join(gender_image_dir, "*")))):
            if debug_max_num_samples is not None and n >= debug_max_num_samples:
                break
            img = cv2.imread(path)
            if img is not None:
                face_roi = detect_face_caffe(img) if detect_face else entire_roi(img)
                if face_roi is None:
                    print("WARNING! No face detected {}".format(path))
                    discarded[gender] += 1
                else:
                    example = {
                        'img': path,
                        'label': category_label,
                        'roi': face_roi,
                        'part': PARTITION_TEST # add support to train/val
                    }
                    if np.max(img) == np.min(img):
                        print('Warning, blank image: %s!' % path)
                    else:
                        data.append(example)
            else:
                print("WARNING! Unable to read %s" % path)
                discarded[gender] += 1
    print("Data loaded. {} samples".format(len(data)))
    print("Discarded {} : {}".format("male", discarded["male"]))
    print("Discarded {} : {}".format("female", discarded["female"]))
    return data


class FERETDatasetGender:
    def __init__(self,
                 partition='test', 
                 imagesdir='gender-feret',
                 target_shape=(256, 256, 3), 
                 augment=False, 
                 custom_augmentation=None, 
                 preprocessing='full_normalization',
                 debug_max_num_samples=None,
                 detect_face=True,
                 cache_dir=None):
        # TODO add support to train/val split
        if not partition.startswith('test'):
            raise Exception("unknown partition")

        self.target_shape = target_shape
        self.custom_augmentation = custom_augmentation
        self.augment = augment
        self.gen = None
        self.preprocessing = preprocessing
        print('Loading %s data...' % partition)

        num_samples = '_' + str(debug_max_num_samples) if debug_max_num_samples is not None else ''
        str_detect_face = "detected" if detect_face else "entire"
        cache_file_name = 'feret_gender_{partition}_{detected}{num_samples}.cache'.format(partition=partition, detected=str_detect_face, num_samples=num_samples) 
        cache_root = os.path.join(EXT_ROOT, CACHE_DIR)

        if cache_dir is None:
            if not os.path.isdir(cache_root): os.mkdir(cache_root)
            cache_file_name = os.path.join(cache_root, cache_file_name)
        else:
            cache_file_name = cache_dir + cache_file_name        

        print("cache file name %s" % cache_file_name)
        try:
            with open(cache_file_name, 'rb') as f:
                print("Loading data from cache", cache_file_name)
                self.data = pickle.load(f)[:debug_max_num_samples]
                print("Data loaded. %d samples, from cache" % (len(self.data)))
        except FileNotFoundError:
            print("Loading %s data from scratch" % partition)

            imagesdir = os.path.abspath(imagesdir)
            imagesdir = os.path.join(imagesdir, "<gender>/<part>_set")
            self.data = _load_feret(imagesdir, partition, debug_max_num_samples, detect_face)
            with open(cache_file_name, 'wb') as f:
                print("Pickle dumping")
                pickle.dump(self.data, f)

    def get_generator(self, batch_size=64, fullinfo=False):
        if self.gen is None:
            self.gen = DataGenerator(self.data, self.target_shape, with_augmentation=self.augment,
                                     custom_augmentation=self.custom_augmentation, batch_size=batch_size,
                                     num_classes=self.get_num_classes(), preprocessing=self.preprocessing,
                                     fullinfo=fullinfo)
        return self.gen

    def get_num_classes(self):
        return NUM_CLASSES

    def get_num_samples(self):
        return len(self.data)


def test1(dataset="test", debug_samples=None):
    dv = FERETDatasetGender(dataset,
                            target_shape=(224, 224, 3),
                            preprocessing='no_normalization',
                            debug_max_num_samples=debug_samples,
                            augment=False)
    print("SAMPLES %d" % dv.get_num_samples())
    print('Now generating from test set')
    gen = dv.get_generator(fullinfo=True)

    i = 0
    while True:
        print(i)
        i += 1
        for n, batch in enumerate(tqdm(gen)):
            for m, (im, gender, path, _) in enumerate(zip(batch[0], batch[1], batch[2], batch[3])):
                print("Sample number", m)

                gender = np.argmax(gender)
                facemax = np.max(im)
                facemin = np.min(im)
                im = (255 * ((im - facemin) / (facemax - facemin))).astype(np.uint8)

                print(im.shape)

                ############################# DEBUG
                # if not os.path.exists("delete_cropped_feret"): os.mkdir("delete_cropped_feret")
                # cv2.imwrite('delete_cropped_feret/{}-{}-raw.jpg'.format(n, m), im)
                ###################################

                cv2.putText(im, "%d %s" % (gender, get_gender_string(gender)), (0, im.shape[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
                cv2.imshow('image {}'.format(n), im)

                ############################# DEBUG
                # cv2.imwrite('delete_cropped_feret/{}-{}.jpg'.format(n, m), im)
                ###################################

                cv2.imshow('image original {}'.format(n), cv2.imread(path))

                if cv2.waitKey(0) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    return


if '__main__' == __name__:
    test1("test")
    print("------LOAD-----")
    test1("test")
