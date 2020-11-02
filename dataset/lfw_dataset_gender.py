from cv2 import cv2
from tqdm import tqdm
import os
import pickle
import numpy as np
import csv
import sys

sys.path.append("../training")
from dataset_tools import DataGenerator

EXT_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = "cache"

DEFAULT_IMAGESDIR = 'data/gender-access/lfw_cropped'
DEFAULT_CSVMETA = 'data/gender-access/lfw_theirs_<gender>.csv'

PARTITION_TEST = 2
NUM_CLASSES = 2

FEMALE_LABEL = 0
MALE_LABEL = 1


def get_gender_label(gender_string):
    if gender_string.startswith("m"):
        return MALE_LABEL
    elif gender_string.startswith("f"):
        return FEMALE_LABEL
    else:
        return None


def _readcsv(csvpath, debug_max_num_samples=None):
    data = []
    with open(csvpath, newline='', encoding="utf8") as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True, delimiter=',', quotechar='|')
        i = 0
        for row in reader:
            if debug_max_num_samples is not None and i >= debug_max_num_samples:
                break
            i = i + 1
            data.append(row)
    return np.array(data)


def _load_lfw_by_gender(csvmeta, imagesdir, gender, debug_max_num_samples=None):
    csvmeta = csvmeta.replace('<gender>', gender)
    meta = _readcsv(csvmeta, debug_max_num_samples)
    print('csv %s read complete: %d.' % (csvmeta, len(meta)))
    data = []
    n_discarded = 0
    for d in tqdm(meta):
        path = os.path.join(imagesdir, '%s' % (d[2]))
        category_label = get_gender_label(gender)
        img = cv2.imread(path)
        if img is not None:
            roi = [0, 0, img.shape[0], img.shape[1]]
            example = {
                'img': path,
                'label': category_label,
                'roi': roi,
                'part': PARTITION_TEST
            }
            if np.max(img) == np.min(img):
                print('Warning, blank image: %s!' % path)
            else:
                data.append(example)
        else:  # img is None
            print("WARNING! Unable to read %s" % path)
            n_discarded += 1
    print("Data loaded. %d samples (%d discarded)" % (len(data), n_discarded))
    return data


def get_gender_string(label):
    if label == MALE_LABEL:
        return "male"
    elif label == FEMALE_LABEL:
        return "female"
    else:
        return label


class LFWPlusDatasetGender:
    def __init__(self,
                partition='test',
                imagesdir=DEFAULT_IMAGESDIR,
                csvmeta=DEFAULT_CSVMETA,
                target_shape=(256, 256, 3),
                augment=True,
                custom_augmentation=None,
                preprocessing='full_normalization',
                debug_max_num_samples=None,
                cache_dir=None):
        if not partition.startswith('test'):
            raise Exception("unknown partition")

        self.target_shape = target_shape
        self.custom_augmentation = custom_augmentation
        self.augment = augment
        self.gen = None
        self.preprocessing = preprocessing
        print('Loading %s data...' % partition)

        num_samples = '_' + str(debug_max_num_samples) if debug_max_num_samples is not None else ''
        cache_file_name = 'lfwplus_gender_{partition}{num_samples}.cache'.format(partition=partition, num_samples=num_samples) 
        cache_root = os.path.join(EXT_ROOT, CACHE_DIR)

        if cache_dir is None:
            if not os.path.isdir(cache_root): os.mkdir(cache_root)
            cache_file_name = os.path.join(cache_root, cache_file_name)
        else:
            cache_file_name = cache_dir + cache_file_name  

        print("cache file name %s" % cache_file_name)
        try:
            with open(cache_file_name, 'rb') as f:
                self.data = pickle.load(f)
                self.data = self.data[:debug_max_num_samples]
                print("Data loaded. %d samples, from cache" % (len(self.data)))
        except FileNotFoundError:
            print("Loading %s data from scratch" % partition)

            imagesdir = os.path.abspath(imagesdir)
            csvmeta = os.path.abspath(csvmeta)

            self.data = _load_lfw_by_gender(csvmeta, imagesdir, "female", debug_max_num_samples)
            self.data += _load_lfw_by_gender(csvmeta, imagesdir, "male", debug_max_num_samples)
            with open(cache_file_name, 'wb') as f:
                print("Pickle dumping")
                pickle.dump(self.data, f)

    def get_generator(self, batch_size=64):
        if self.gen is None:
            self.gen = DataGenerator(self.data, self.target_shape, with_augmentation=self.augment,
                                     custom_augmentation=self.custom_augmentation, batch_size=batch_size,
                                     num_classes=self.get_num_classes(), preprocessing=self.preprocessing)
        return self.gen

    def get_num_classes(self):
        return NUM_CLASSES

    def get_num_samples(self):
        return len(self.data)


def test1(dataset="test", debug_samples=None):
    dv = LFWPlusDatasetGender(dataset, target_shape=(224, 224, 3), preprocessing='full_normalization',
                              debug_max_num_samples=debug_samples, augment=False)
    print("SAMPLES %d" % dv.get_num_samples())
    print('Now generating from test set')
    gen = dv.get_generator()

    i = 0
    while True:
        print(i)
        i += 1
        for batch in tqdm(gen):
            for im, gender in zip(batch[0], batch[1]):
                gender = np.argmax(gender)
                facemax = np.max(im)
                facemin = np.min(im)
                im = (255 * ((im - facemin) / (facemax - facemin))).astype(np.uint8)
                cv2.putText(im, "%d %s" % (gender, get_gender_string(gender)), (0, im.shape[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
                cv2.imshow('vggface2 image', im)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    return


if '__main__' == __name__:
    test1("test")
    print("------LOAD-----")
    test1("test")
