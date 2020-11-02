# import warnings;
# warnings.filterwarnings('ignore', category=FutureWarning)

from cv2 import cv2
from tqdm import tqdm
import os
import pickle
import numpy as np
import csv
import sys
from glob import glob

sys.path.append("../training")
from dataset_tools import DataGenerator, VGGFACE2_MEANS, mean_std_normalize

EXT_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = "cache"
DATA_DIR = "data"

PARTITION_TEST = 2
NUM_CLASSES = 2

FEMALE_LABEL = 0
MALE_LABEL = 1

# mivia_tree = {
#     'unisa_public' : {
#         "<gender>" : {
#             "<partition>_set" : {
#                 "opencv" : 'b/n'
#             }
#         }
#     },
#     'unisa_private' : {
#         "<gender>" :{
#             "<partition>_set": {
#                 '<partition>_set_0' : 'b/n',
#                 '<partition>_set_1' : 'color',
#                 '<partition>_set_2' : 'color',
#                 '<partition>_set_3' : 'color',
#                 '<partition>_set_4' : 'color',
#                 '<partition>_set_5' : 'color',
#             }
#         }
#     }
# }

mivia_tree = [
    ('unisa_private/female/test_set/test_set_0', 'b/n'),
    ('unisa_private/female/test_set/test_set_1', 'color'),
    ('unisa_private/female/test_set/test_set_2', 'color'),
    ('unisa_private/female/test_set/test_set_3', 'color'),
    ('unisa_private/female/test_set/test_set_4', 'color'),
    ('unisa_private/female/test_set/test_set_5', 'color'),
    ('unisa_private/female/training_set/training_set_0', 'b/n'),
    ('unisa_private/female/training_set/training_set_1', 'color'),
    ('unisa_private/female/training_set/training_set_2', 'color'),
    ('unisa_private/female/training_set/training_set_3', 'color'),
    ('unisa_private/female/training_set/training_set_4', 'color'),
    ('unisa_private/female/training_set/training_set_5', 'color'),
    ('unisa_private/male/test_set/test_set_0', 'b/n'),
    ('unisa_private/male/test_set/test_set_1', 'color'),
    ('unisa_private/male/test_set/test_set_2', 'color'),
    ('unisa_private/male/test_set/test_set_3', 'color'),
    ('unisa_private/male/test_set/test_set_4', 'color'),
    ('unisa_private/male/test_set/test_set_5', 'color'),
    ('unisa_private/male/training_set/training_set_0', 'b/n'),
    ('unisa_private/male/training_set/training_set_1', 'color'),
    ('unisa_private/male/training_set/training_set_2', 'color'),
    ('unisa_private/male/training_set/training_set_3', 'color'),
    ('unisa_private/male/training_set/training_set_4', 'color'),
    ('unisa_private/male/training_set/training_set_5', 'color'),
    ('unisa_public/female/test_set/opencv', 'b/n'),
    ('unisa_public/female/training_set/opencv', 'b/n'),
    ('unisa_public/male/test_set/opencv', 'b/n'),
    ('unisa_public/male/training_set/opencv', 'b/n'),
]


def _get_dataset_imagesdir(partition, gender):
    for tmp_path, _ in mivia_tree:
        _, tmp_gender, tmp_partition, _ = tmp_path.split('/')
        if tmp_gender == gender and partition in tmp_partition:
            yield tmp_path


def mkdir_recursive(tree):
    os.makedirs(tree, exist_ok=True)


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

def _no_normalization(inpath, outpath):
    for inimg in tqdm(sorted(glob(os.path.join(inpath, "*")))):
        inimg_name = os.path.split(inimg)[-1]
        outimg = cv2.imread(inimg)
        if (len(outimg.shape)<3 or outimg.shape[2]<3):
            outimg = np.repeat(np.squeeze(outimg)[:,:,None], 3, axis=2)
        cv2.imwrite(os.path.join(outpath, inimg_name), outimg)


def _vggface2_normalization(inpath, outpath):
    ds_means = VGGFACE2_MEANS
    ds_stds = None
    for inimg_path in tqdm(sorted(glob(os.path.join(inpath, "*")))):
        inimg_name = os.path.split(inimg_path)[-1]
        inimg = cv2.imread(inimg_path)

        outimg = mean_std_normalize(inimg, ds_means, ds_stds)

        facemax = np.max(outimg)
        facemin = np.min(outimg)
        outimg = (255 * ((outimg - facemin) / (facemax - facemin))).astype(np.uint8)
        
        if (len(outimg.shape)<3 or outimg.shape[2]<3):
            outimg = np.repeat(np.squeeze(outimg)[:,:,None], 3, axis=2)
        cv2.imwrite(os.path.join(outpath, inimg_name), outimg)


def _mivia_gender_normalize(inpath, outpath):
    for i, (path, color) in enumerate(mivia_tree):
        tmp_inpath = os.path.join(inpath, path)
        tmp_outpath = os.path.join(outpath, path)
        print("Normalization {} {}/{}...".format(color, i+1, len(mivia_tree)))
        print(tmp_outpath)
        mkdir_recursive(tmp_outpath)
        if color == 'b/n':
            _no_normalization(tmp_inpath, tmp_outpath)
        else:
            _vggface2_normalization(tmp_inpath, tmp_outpath)
        
        original_samples = len(os.listdir(tmp_inpath))
        normalized_samples = len(os.listdir(tmp_outpath))
        print("Original samples", original_samples)
        print("Normalized samples", normalized_samples)
        assert original_samples == normalized_samples, "{} samples not processed".format(original_samples - normalized_samples)

    return outpath


def entire_roi(img):
    return [0, 0, img.shape[1], img.shape[0]]


def _load_mivia_gender(imagesdir, partition="test", debug_max_num_samples=None):
    data = list()
    discarded = {"male":0, "female":0}
    
    if partition.startswith("train") or partition.startswith("val"):
        dir_partition = "training"
    else:
        dir_partition = partition

    for gender in ["male", "female"]:
        for gender_image_dir in _get_dataset_imagesdir(dir_partition, gender):
            gender_image_dir = os.path.join(imagesdir, gender_image_dir)
            print(gender_image_dir)
            category_label = get_gender_label(gender)
            for n, path in enumerate(tqdm(sorted(glob(os.path.join(gender_image_dir, "*"))))):
                if debug_max_num_samples is not None and n >= debug_max_num_samples:
                    break
                img = cv2.imread(path)
                if img is not None:
                    example = {
                        'img': path,
                        'label': category_label,
                        'roi': entire_roi(img),
                        'part': PARTITION_TEST # add support to train/val
                    }
                    if np.max(img) == np.min(img):
                        print('Warning, blank image: %s!' % path)
                        discarded[gender] += 1
                    else:
                        data.append(example)
                else:
                    print("WARNING! Unable to read %s" % path)
                    discarded[gender] += 1
    print("Data loaded. {} samples".format(len(data)))
    print("Discarded {} : {}".format("male", discarded["male"]))
    print("Discarded {} : {}".format("female", discarded["female"]))
    return data


class MIVIADatasetGender:
    def __init__(self,
                 partition='test', 
                 imagesdir='mivia-gender',
                 target_shape=(256, 256, 3), 
                 augment=False, 
                 custom_augmentation=None, 
                 preprocessing='no_normalization',
                 debug_max_num_samples=None,
                 prenormalized=True):
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
        str_normalized = "_prenormalized" if prenormalized else ""
        cache_file_name = 'mivia_gender{normalized}_{partition}{num_samples}.cache'.format(normalized=str_normalized,partition=partition, num_samples=num_samples)

        cache_root = os.path.join(EXT_ROOT, CACHE_DIR)
        if not os.path.isdir(cache_root): os.mkdir(cache_root)
        cache_file_name = os.path.join(cache_root, cache_file_name)
        print("cache file name %s" % cache_file_name)
        try:
            with open(cache_file_name, 'rb') as f:
                print("Loading data from cache", cache_file_name)
                self.data = pickle.load(f)[:debug_max_num_samples]
                print("Data loaded. %d samples, from cache" % (len(self.data)))
        except FileNotFoundError:
            print("Loading %s data from scratch" % partition)
            
            images_root = os.path.join(EXT_ROOT, DATA_DIR)
            imagesdir = os.path.join(images_root, imagesdir)
            prenormalized_imagesdir = imagesdir + '-normalized'

            if prenormalized:
                if not os.path.exists(prenormalized_imagesdir) or not len(os.listdir(prenormalized_imagesdir)):
                    imagesdir = _mivia_gender_normalize(imagesdir, prenormalized_imagesdir)
                imagesdir = prenormalized_imagesdir
            
            self.data = _load_mivia_gender(imagesdir, partition, debug_max_num_samples)
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
    dv = MIVIADatasetGender(dataset,
                            target_shape=(224, 224, 3),
                            preprocessing='no_normalization',
                            debug_max_num_samples=debug_samples,
                            augment=False,)
    print("SAMPLES %d" % dv.get_num_samples())
    print('Now generating from test set')
    gen = dv.get_generator(fullinfo=True)

    # export = True

    i = 0
    while True:
        print(i)
        i += 1
        for n, batch in enumerate(tqdm(gen)):
            for m, (im, gender, path, _) in enumerate(zip(batch[0], batch[1], batch[2], batch[3])):
                gender = np.argmax(gender)
                facemax = np.max(im)
                facemin = np.min(im)
                im = (255 * ((im - facemin) / (facemax - facemin))).astype(np.uint8)

                counter = n*m+m
                print("sequence", counter)
                print(im.shape, path)

                ############################# DEBUG
                # if not os.path.exists("delete_cropped_feret"): os.mkdir("delete_cropped_feret")
                # cv2.imwrite('delete_cropped_feret/{}-{}-raw.jpg'.format(n, m), im)
                ###################################

                cv2.putText(im, "%d %s" % (gender, get_gender_string(gender)), (0, im.shape[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
                cv2.imshow('image', im)

                ############################# DEBUG
                # cv2.imwrite('delete_cropped_feret/{}-{}.jpg'.format(n, m), im)
                ###################################

                im_original = cv2.imread(path)
                print(im_original.shape, path)
                cv2.imshow('image original', im_original)

                before_preprocessing = path.replace('mivia-gender-normalized', 'mivia-gender')
                before_preprocessing_im = cv2.imread(before_preprocessing)
                print(before_preprocessing_im.shape, before_preprocessing)
                cv2.imshow('image before_preprocessing', before_preprocessing_im)

                if cv2.waitKey(0) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    return

def test2(dataset="test", debug_samples=None):
    dv = MIVIADatasetGender(dataset,
                            target_shape=(224, 224, 3),
                            preprocessing='vggface2',
                            debug_max_num_samples=debug_samples,
                            augment=False,
                            prenormalized=False)
    print("SAMPLES %d" % dv.get_num_samples())
    print('Now generating from test set')
    gen = dv.get_generator(fullinfo=True)

    # export = True

    i = 0
    while True:
        print(i)
        i += 1
        for n, batch in enumerate(tqdm(gen)):
            for m, (im, gender, path, _) in enumerate(zip(batch[0], batch[1], batch[2], batch[3])):
                gender = np.argmax(gender)
                facemax = np.max(im)
                facemin = np.min(im)

                im = (255 * ((im - facemin) / (facemax - facemin))).astype(np.uint8)

                counter = n*m+m
                print("sequence", counter)
                print(im.shape, path)

                cv2.putText(im, "%d %s" % (gender, get_gender_string(gender)), (0, im.shape[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
                cv2.imshow('image', im)

                im_original = cv2.imread(path)
                print(im_original.shape, path)
                cv2.imshow('image original', im_original)

                if cv2.waitKey(0) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    return


if '__main__' == __name__:
    test1("test")
    print("------LOAD-----")
    test1("test")
