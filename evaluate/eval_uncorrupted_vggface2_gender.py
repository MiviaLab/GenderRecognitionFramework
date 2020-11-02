import keras
import sys, os, re
from glob import glob
from datetime import datetime
import time

sys.path.append('../training/scratch_models')
from mobile_net_v2_keras import relu6

sys.path.append('../training/keras_vggface/keras_vggface')
from antialiasing import BlurPool

sys.path.append('../dataset')
from vgg2_dataset_gender import Vgg2DatasetGender as Dataset

import argparse

parser = argparse.ArgumentParser(description='VggFace2 evaluation, provided for train and val partition')
parser.add_argument('--path', dest='inpath', type=str, help='source path of model to test')
parser.add_argument('--gpu', dest="gpu", type=str, default="0", help="gpu to use")
parser.add_argument('--outf', dest="outf", type=str, default="results/vggface2", help='destination path of results file')
parser.add_argument('--time', action='store_true')
args = parser.parse_args()


def load_keras_model(filepath):
    model = keras.models.load_model(filepath, custom_objects={'BlurPool': BlurPool, 'relu6': relu6})
    if 'mobilenet96' in filepath:
        INPUT_SHAPE = (96, 96, 3)
    elif 'mobilenet64_bio' in filepath:
        INPUT_SHAPE = (64, 64, 3)
    elif 'xception71' in filepath:
        INPUT_SHAPE = (71, 71, 3)
    elif 'xception' in filepath:
        INPUT_SHAPE = (299, 299, 3)
    else:
        INPUT_SHAPE = (224, 224, 3)
    return model, INPUT_SHAPE


ep_re = re.compile('checkpoint.([0-9]+).hdf5')


def _find_latest_checkpoint(d):
    all_checks = glob(os.path.join(d, '*'))
    max_ep = 0
    max_c = None
    for c in all_checks:
        epoch_num = re.search(ep_re, c)
        if epoch_num is not None:
            epoch_num = int(epoch_num.groups(1)[0])
            if epoch_num > max_ep:
                max_ep = epoch_num
                max_c = c
    return max_c


def get_allchecks(dirpath):
    alldirs = glob(os.path.join(dirpath, '*'))
    allchecks = [_find_latest_checkpoint(d) for d in alldirs]
    return [c for c in allchecks if c is not None]


def evalds(filepath, outf_path, partition, batch_size=64):
    print('Partition: %s' % partition)
    outf = open(outf_path, "a+")
    outf.write('Results for: %s\n' % filepath)
    model, INPUT_SHAPE = load_keras_model(filepath)

    dataset_test = Dataset(partition, target_shape=INPUT_SHAPE, augment=False, preprocessing='vggface2')

    data_gen = dataset_test.get_generator(batch_size)
    print("Dataset batches %d" % len(data_gen))
    start_time = time.time()
    result = model.evaluate_generator(data_gen, verbose=1, workers=4)
    spent_time = time.time() - start_time
    batch_average_time = spent_time / len(data_gen)
    print("Evaluate time %d s" % spent_time)
    print("Batch time %.10f s" % batch_average_time)
    o = "%s %f\n" % (partition, result[1])
    print("\n\n RES " + o)
    outf.write(o)

    outf.write('\n\n')
    outf.close()


def evalds_all(dirpath, outf_path, partition='test'):
    allchecks = get_allchecks(dirpath)
    for c in allchecks:
        print('\n Testing %s now...\n' % c)
        evalds(c, outf_path=outf_path, partition=partition)



if '__main__' == __name__:
    start_time = datetime.today()
    os.makedirs(args.outf, exist_ok=True)

    out_path = os.path.join(args.outf, "results.txt")

    if args.time:
        out_path = "%s_%s%s" % (
            os.path.splitext(out_path)[0], start_time.strftime('%Y%m%d_%H%M%S'), os.path.splitext(out_path)[1])
        print("Exporting to %s" % out_path)

    if args.gpu is not None:
        gpu_to_use = [str(s) for s in args.gpu.split(',') if s.isdigit()]
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_to_use)
        print("WARNING: Using GPU %s" % os.environ["CUDA_VISIBLE_DEVICES"])

    for partition in ['train', 'val']:
        if args.inpath.endswith('.hdf5'):
            evalds(args.inpath, outf_path=out_path, partition=partition)
        else:
            evalds_all(args.inpath, outf_path=out_path, partition=partition)

    print("GPU execution time: %s" % str(datetime.today() - start_time))
