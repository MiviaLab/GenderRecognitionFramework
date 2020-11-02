#!/usr/bin/python3
from cv2 import cv2
from tqdm import tqdm
import os
import pickle
import numpy as np
import csv
import sys

from vgg2_utils import get_id_from_vgg2, PARTITION_TEST, PARTITION_VAL, PARTITION_TRAIN

sys.path.append("../training")
from dataset_tools import enclosing_square, add_margin, DataGenerator, VGGFace2Augmentation

EXT_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = "cache"
DATA_DIR = "data"

NUM_CLASSES = 2

FEMALE_LABEL = 0
MALE_LABEL = 1

vgg2gender = None
gender2vgg = None


def _load_identities(idmetacsv):
    global vgg2gender
    global gender2vgg
    if vgg2gender is None:
        vgg2gender = {}
        gender2vgg = []
        arr = _readcsv(idmetacsv)
        i = 0
        for line in arr:
            try:
                vggnum = int(line[0][1:])
                vgg2gender[vggnum] = (get_gender_label(line[-1]), i)
                gender2vgg.append((get_gender_label(line[-1]), vggnum))
                i += 1
            except ValueError:
                pass
        print(len(gender2vgg), len(vgg2gender), NUM_CLASSES)


def get_gender_label(gender_letter):
    if gender_letter == 'm':
        return MALE_LABEL
    elif gender_letter == 'f':
        return FEMALE_LABEL
    else:
        print("Error gender deserialize")
        return None


def get_gender_string(label):
    if label == MALE_LABEL:
        return "male"
    elif label == FEMALE_LABEL:
        return "female"
    else:
        return label


def get_gender_from_vgg2(vggidn, idmetacsv='vggface2/identity_meta.csv'):
    _load_identities(idmetacsv)
    try:
        return vgg2gender[vggidn]
    except KeyError:
        print('ERROR: n%d unknown' % vggidn)
        return 'unknown', -1


def get_vgg2_gender(idn, idmetacsv='vggface2/identity_meta.csv'):
    _load_identities(idmetacsv)
    try:
        return gender2vgg[idn]
    except IndexError:
        print('ERROR: %d unknown', idn)
        return 'unknown', -1


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


def _load_vgg2(csvmeta, imagesdir, partition, debug_max_num_samples=None):
    imagesdir = imagesdir.replace('<part>', partition)
    csvmeta = csvmeta.replace('<part>', partition)
    meta = _readcsv(csvmeta, debug_max_num_samples)
    print('csv %s read complete: %d.' % (csvmeta, len(meta)))
    idmetacsv = os.path.join(os.path.dirname(csvmeta), 'identity_meta.csv')
    data = []
    n_discarded = 0
    for d in tqdm(meta):
        _, category_label = get_id_from_vgg2(int(d[3]), idmetacsv)
        sub_category_label, _ = get_gender_from_vgg2(int(d[3]), idmetacsv)
        path = os.path.join(imagesdir, '%s' % (d[2]))
        img = cv2.imread(path)
        roi = [int(x) for x in d[4:8]]
        roi = enclosing_square(roi)
        # roi = add_margin(roi, 0.2)
        if partition.startswith("train") or partition.startswith('val'):
            sample_partition = get_partition(category_label, sub_category_label)
        else:
            sample_partition = PARTITION_TEST

        if img is not None:
            example = {
                'img': path,
                'label': sub_category_label,
                'roi': roi,
                'part': sample_partition
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


people_by_gender = {
    FEMALE_LABEL: dict(),
    MALE_LABEL: dict()
}


def get_partition(identity_label, gender_label):
    if gender_label == MALE_LABEL:
        return split_by_identity(MALE_LABEL, identity_label)
    elif gender_label == FEMALE_LABEL:
        return split_by_identity(FEMALE_LABEL, identity_label)
    else:
        return None


def split_by_identity(gender_label, identity_label):
    global people_by_gender
    try:
        faces, partition = people_by_gender[gender_label][identity_label]
        people_by_gender[gender_label][identity_label] = (faces + 1, partition)
    except KeyError:
        l = len(people_by_gender[gender_label])
        # split 10/90 stratified by identity
        l = (l - 1) % 10
        if l == 0:
            partition = PARTITION_VAL
        else:
            partition = PARTITION_TRAIN
        people_by_gender[gender_label][identity_label] = (1, partition)
    return partition


class Vgg2DatasetGender:
    def __init__(self,
                partition='train',
                imagesdir='vggface2_data/<part>',
                csvmeta='vggface2_data/annotations/<part>.detected.csv',
                target_shape=(224, 224, 3),
                augment=True,
                custom_augmentation=None,
                preprocessing='full_normalization',
                debug_max_num_samples=None):
        if partition.startswith('train'):
            partition_label = PARTITION_TRAIN
        elif partition.startswith('val'):
            partition_label = PARTITION_VAL
        elif partition.startswith('test'):
            partition_label = PARTITION_TEST
        else:
            raise Exception("unknown partition")

        self.target_shape = target_shape
        self.custom_augmentation = custom_augmentation
        self.augment = augment
        self.gen = None
        self.preprocessing = preprocessing
        print('Loading %s data...' % partition)

        num_samples = '_' + str(debug_max_num_samples) if debug_max_num_samples is not None else ''
        cache_file_name = 'vggface2_gender_{partition}{num_samples}.cache'.format(partition=partition, num_samples=num_samples)   

        cache_root = os.path.join(EXT_ROOT, CACHE_DIR)
        if not os.path.isdir(cache_root): os.mkdir(cache_root)
        cache_file_name = os.path.join(cache_root, cache_file_name)

        print("cache file name %s" % cache_file_name)
        try:
            with open(cache_file_name, 'rb') as f:
                self.data = pickle.load(f)
                self.data = self.data[:debug_max_num_samples]
                print("Data loaded. %d samples, from cache" % (len(self.data)))
        except FileNotFoundError:
            print("Loading %s data from scratch" % partition)

            images_root = os.path.join(EXT_ROOT, DATA_DIR)

            csvmeta = os.path.join(images_root, csvmeta)
            imagesdir = os.path.join(images_root, imagesdir)

            load_partition = "train" if partition_label == PARTITION_TRAIN or partition_label == PARTITION_VAL else "test"
            loaded_data = _load_vgg2(csvmeta, imagesdir, load_partition, debug_max_num_samples)
            if partition.startswith('test'):
                self.data = loaded_data
            else:
                self.data = [x for x in loaded_data if x['part'] == partition_label]
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
    global people_by_gender
    if dataset.startswith("train") or dataset.startswith("val"):
        print(dataset, debug_samples if debug_samples is not None else '')
        dt = Vgg2DatasetGender(dataset, target_shape=(224, 224, 3), preprocessing='vggface2',
                               custom_augmentation=VGGFace2Augmentation(), debug_max_num_samples=debug_samples)
        print("SAMPLES %d" % dt.get_num_samples())

        if len(people_by_gender[MALE_LABEL]):
            print("Males %d" % (len(people_by_gender[MALE_LABEL])))
            samples = [v[0] for k, v in people_by_gender[MALE_LABEL].items() if v[1] == PARTITION_TRAIN]
            print("Male samples in train %d (people %d)" % (sum(samples), len(samples)))
            samples = [v[0] for k, v in people_by_gender[MALE_LABEL].items() if v[1] == PARTITION_VAL]
            print("Male samples in validation %d (people %d)" % (sum(samples), len(samples)))

        if len(people_by_gender[FEMALE_LABEL]):
            print("Females %d" % (len(people_by_gender[FEMALE_LABEL])))
            samples = [v[0] for k, v in people_by_gender[FEMALE_LABEL].items() if v[1] == PARTITION_TRAIN]
            print("Female samples in train %d (people %d)" % (sum(samples), len(samples)))
            samples = [v[0] for k, v in people_by_gender[FEMALE_LABEL].items() if v[1] == PARTITION_VAL]
            print("Female samples in validation %d (people %d)" % (sum(samples), len(samples)))

        print('Now generating from %s set' % dataset)
        gen = dt.get_generator()
    else:
        dv = Vgg2DatasetGender('test', target_shape=(224, 224, 3), preprocessing='full_normalization',
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
    test1("train")
    test1("val")
    test1("test")
    
    test1("train") # cache
    test1("val") # cache
    test1("test") # cache
