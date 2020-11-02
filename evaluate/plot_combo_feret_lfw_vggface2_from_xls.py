import argparse
from xls_models_tools import mean_dict, extract_results_by_corruption
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict, OrderedDict
from tabulate import tabulate
from matplotlib.lines import Line2D

from plot_and_tabulate_feret_from_xls import average_by_levels, nine_models_order, compile_chart, blur_labels, noise_labels, digital_labels

parser = argparse.ArgumentParser(description='corruption error calculation')
parser.add_argument('--corrupted', dest='corrupted', type=str, help='corrupted experiment results (.xls)')
parser.add_argument('--uncorrupted', dest='uncorrupted', type=str, help='original experiment results (.xls)')
parser.add_argument('--out', dest='filepath', type=str,  default="results/feret", help='output file path of the chart')
parser.add_argument('--title', dest='title', type=str, default="", help='title of the chart') # Gender recognition accuracy by model
args = parser.parse_args()

# TODO read VGGFACE2 results from file
"""
+----------------+-------------------------+-----------------------+
| Method         |   VGGFACE2-GENDER-TRAIN |   VGGFACE2-GENDER-VAL |
+================+=========================+=======================+
| VGG-16         |                   0.999 |                 0.982 |
+----------------+-------------------------+-----------------------+
| SE-ResNet-50   |                   0.989 |                 0.982 |
+----------------+-------------------------+-----------------------+
| DenseNet-121   |                   0.988 |                 0.981 |
+----------------+-------------------------+-----------------------+
| MobileNet v2-A |                   0.987 |                 0.98  |
+----------------+-------------------------+-----------------------+
| MobileNet v2-B |                   0.983 |                 0.979 |
+----------------+-------------------------+-----------------------+
| MobileNet v2-C |                   0.972 |                 0.971 |
+----------------+-------------------------+-----------------------+
| ShuffleNet     |                   0.986 |                 0.98  |
+----------------+-------------------------+-----------------------+
| SqueezeNet     |                   0.954 |                 0.955 |
+----------------+-------------------------+-----------------------+
| XceptionNet    |                   0.984 |                 0.972 |
+----------------+-------------------------+-----------------------+
"""

VGGFace2_train = {
    "vgg16": 0.999,
    "senet50": 0.989,
    "densenet121bc": 0.988,
    "mobilenet224": 0.987,
    "mobilenet96": 0.983,
    "mobilenet64": 0.972,
    "shufflenet224": 0.986,
    "squeezenet": 0.954,
    "xception71": 0.984,
}

VGGFace2_val = {
    "vgg16": 0.982,
    "senet50": 0.982,
    "densenet121bc": 0.981,
    "mobilenet224": 0.980,
    "mobilenet96": 0.979,
    "mobilenet64": 0.971,
    "shufflenet224": 0.980,
    "squeezenet": 0.955,
    "xception71": 0.972,
}

"""
+----------------+--------+---------+
| Method         |   LFW+ |   LFW+C |
+================+========+=========+
| VGG-16         |  0.989 |   0.963 |
+----------------+--------+---------+
| SE-ResNet-50   |  0.989 |   0.968 |
+----------------+--------+---------+
| DenseNet-121   |  0.985 |   0.955 |
+----------------+--------+---------+
| MobileNet v2-A |  0.984 |   0.958 |
+----------------+--------+---------+
| MobileNet v2-B |  0.983 |   0.941 |
+----------------+--------+---------+
| MobileNet v2-C |  0.966 |   0.929 |
+----------------+--------+---------+
| ShuffleNet     |  0.983 |   0.946 |
+----------------+--------+---------+
| SqueezeNet     |  0.948 |   0.906 |
+----------------+--------+---------+
| XceptionNet    |  0.957 |   0.923 |
+----------------+--------+---------+
"""
# TODO read LFW results from file
LFW_original = {
    "vgg16": 0.989,
    "senet50": 0.989,
    "densenet121bc": 0.985,
    "mobilenet224": 0.984,
    "mobilenet96": 0.983,
    "mobilenet64": 0.966,
    "shufflenet224": 0.983,
    "squeezenet": 0.948,
    "xception71": 0.957,
}

LFW_corrupted = {
    "vgg16": 0.963,
    "senet50": 0.968,
    "densenet121bc": 0.955,
    "mobilenet224": 0.958,
    "mobilenet96": 0.941,
    "mobilenet64": 0.929,
    "shufflenet224": 0.946,
    "squeezenet": 0.906,
    "xception71": 0.923,
}


official_labels_vgg_feret = {
    "vgg-train": "VGGFace2 train",
    "vgg-val": "VGGFace2 validation",
    "LFW" : "LFW+",
    "LFW-C" : "LFW+C",
    "FERET": "GF",
    "FERET-C": "GF-C",
}

fantasy = ("*", "..", "xx", "\\", "++", "//", "||", "--", "o", "O")
colors = ("darkcyan", "mediumorchid", "crimson", "black", 'darkslateblue', 'sandybrown',
          'cornflowerblue', 'lightsalmon', 'royalblue', 'darkolivegreen', 'chocolate', 'lightseagreen')


def create_chart_models(data_labels, models_dict, save_file_path='test.png', title=''):
    compress = False
    model_labels = list(models_dict.keys())
    x = np.arange(len(model_labels))
    # width = 1.5 / (len(data_labels) + 4)

    # lfw
    width = 1.3 / (len(data_labels) + 5)

    offset = (len(data_labels) - 1) / 2  # (len(corruption_labels) - 1) / 2
    data_dict = defaultdict(list)
    for i, lab in enumerate(data_labels):
        data_dict[lab].extend([model_values[i] for model_values in models_dict.values()])
    # if order_and_rename:
    #     data_dict = {official_labels[k]: v for k, v in data_dict.items()}
    #     keyorder = {k: v for v, k in enumerate(official_labels.values())}
    #     data_dict = OrderedDict(sorted(data_dict.items(), key=lambda i: keyorder.get(i[0])))
    ncol = len(data_labels)

    # lfw
    # ncol = 3

    #lfw
    compress = True

    art = compile_chart(data_dict, width, title, x, offset, model_labels, ncol, patterns=fantasy, colors=colors, compress=compress)
    # same_color=('white', 'indianred'))

    plt.savefig(save_file_path, additional_artists=art, bbox_inches="tight", dpi=300)


def plot_bar_chart(corrupted_exp, uncorrupted_exp, filepath, title, by_category=False, debug=False):
    data_means = defaultdict(dict)
    data_means_compress = defaultdict(list)

    corruptions = list(official_labels_vgg_feret.values())

    data_means = average_by_levels(corrupted_exp, data_means)

    # append vgg-train
    for model, vgg_train in VGGFace2_train.items():
        data_means_compress[model].append(vgg_train)

    # append vgg-val
    for model, vgg_val in VGGFace2_val.items():
        data_means_compress[model].append(vgg_val)

    # append lfw original
    for model, lfw_original_value in LFW_original.items():
        data_means_compress[model].append(lfw_original_value)

    # append lfw corrupted
    for model, lfw_corrupted_value in LFW_corrupted.items():
        data_means_compress[model].append(lfw_corrupted_value)

    # append FERET
    uncorr_data = next(iter(uncorrupted_exp.values()))
    for model, uncorr_value in uncorr_data.items():
        data_means_compress[model].append(uncorr_value)

    # append FERET-C
    for model, corr_dict in data_means.items():
        tmp_list = list()
        for corr_key, corr_mean in corr_dict.items():
            # tmp_list.append(corr_mean)
            if corr_key in blur_labels or corr_key in noise_labels or corr_key in digital_labels:
                tmp_list.append(corr_mean)
        data_means_compress[model].append(sum(tmp_list) / len(tmp_list))

    data_means_compress = nine_models_order(data_means_compress)

    create_chart_models(corruptions, data_means_compress, filepath, title)


if __name__ == '__main__':
    corrupted_results = extract_results_by_corruption(args.corrupted)
    uncorrupted_results = extract_results_by_corruption(args.uncorrupted)
    plot_bar_chart(corrupted_results, uncorrupted_results, os.path.join(args.filepath, "combo"), args.title)
