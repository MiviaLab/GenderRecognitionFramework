# Gender recognition in the wild: a robustness evaluation over corrupted images

This repository contains the code for the paper *Gender recognition in the wild: a robustness evaluation over corrupted images - A. Greco, A. Saggese, M. Vento, V. Vigilante - Journal of Ambient Intelligence and Humanized Computing 2020*

If you use this code in your research, please cite this paper.


Gender recognition framework provided by [MIVIA Lab](https://mivia.unisa.it), including training and evaluation code.

The repository includes the code for generating the corrupted version of the LFW+ dataset (LFW+C) as well as other corrupted datasets, in order to allow the evaluation of the model robustness to image corruptions.




## Setup

Python3 libraries you need to install:

```
numpy==1.18.4
opencv-python==4.2.0.34
tensorflow-gpu==1.14.0
Keras==2.3.1
matplotlib==3.2.1
Pillow==7.1.2
scikit-image==0.17.2
scipy==1.4.1
tabulate==0.8.7
tqdm==4.46.1
```

You will also need to download all the data for the datasets that you intend to use and extract it in the `/dataset/data` directory.
You will find the annotation for vggface2 which includes the detected regions with the faces [here](https://github.com/MiviaLab/GenderRecognitionFramework/releases/tag/0), you will need to download the images separately from the official website.

## Dataset
The implemented _datasets_ are VGGFACE2, LFW+, MIVIA and FERET. <br>
Run these commands from dataset directory in order to test them:

```bash
python3 vgg2_dataset_gender.py
python3 lfw_dataset_gender.py
python3 mivia_dataset_gender.py
python3 feret_dataset_gender.py
```

## Corrupted images dataset

In order to export dataset augmented with corruptions, run these commands from _dataset_ directory:

```bash
python3 lfw_plus_aug_dataset.py exp
python3 feret_aug_dataset.py exp
```

## Train
In order to train neural networks, you must run <code>train.py</code> script from the _training_ directory.<br>
Here the used commands to train the associated paper solutions.

```bash
python3 train.py --net squeezenet --dataset vggface2_gender --pretraining imagenet --preprocessing vggface2 --augmentation default --batch 128 --lr 0.005:0.2:20 --sel_gpu 0 --training-epochs 70 --weight_decay 0.005 --momentum
```
```bash
python3 train.py --net shufflenet224 --dataset vggface2_gender --pretraining imagenet --preprocessing vggface2 --augmentation default --batch 128 --lr 0.005:0.2:20 --sel_gpu 0 --training-epochs 70 --weight_decay 0.005 --momentum
```
```bash
python3 train.py --net mobilenet224 --dataset vggface2_gender --pretraining imagenet --preprocessing vggface2 --augmentation default --batch 128 --lr 0.005:0.2:20 --sel_gpu 0 --training-epochs 70 --weight_decay 0.005 --momentum
```
```bash
python3 train.py --net mobilenet96 --dataset vggface2_gender --pretraining imagenet --preprocessing vggface2 --augmentation default --batch 128 --lr 0.005:0.2:20  --sel_gpu 0 --training-epochs 70 --weight_decay 0.005 --momentum
```
```bash
python3 train.py --net mobilenet64_bio --dataset vggface2_gender --preprocessing vggface2 --augmentation default --batch 128 --lr 0.005:0.2:20 --sel_gpu 0 --training-epochs 70 --weight_decay 0.005 --momentum
```
```bash
python3 train.py --net xception71 --dataset vggface2_gender --pretraining imagenet --preprocessing vggface2 --augmentation default --batch 128 --lr 0.005:0.2:20 --sel_gpu 0 --training-epochs 70 --weight_decay 0.005 --momentum
```
```bash
python3 train.py --net senet50 --dataset vggface2_gender --pretraining imagenet --preprocessing vggface2 --augmentation default --batch 128 --lr 0.005:0.2:20 --sel_gpu 0 --training-epochs 70 --weight_decay 0.005 --momentum
```
```bash
python3 train.py --net densenet121bc --dataset vggface2_gender --pretraining imagenet --preprocessing vggface2 --augmentation default --batch 128 --lr 0.005:0.2:20 --sel_gpu 0 --training-epochs 70 --weight_decay 0.005 --momentum
```
```bash
python3 train.py --net vgg16 --dataset vggface2_gender --pretraining vggface2 --preprocessing vggface2 --augmentation default --batch 128 --lr 0.005:0.2:20 --sel_gpu 0 --training-epochs 70 --weight_decay 0.005 --momentum
```
```bash
python3 train.py --net vgg16 --dataset vggface2_gender --pretraining imagenet --preprocessing vggface2 --augmentation default --batch 128 --lr 0.005:0.2:20 --sel_gpu 0,1,2 --ngpus 3 --training-epochs 70 --weight_decay 0.005 --momentum
```

## Evaluation
In order to evaluate the networks, move into the _evaluate_ directory and run the following commands according to the dataset you want to test which. In the subdirectory _results_, as the name suggests, you will find the results of these scripts, divided by dataset.

For each dataset, the provided commands must be executed in order beacuse each command depends on the results of the previous ones.

In _combo_ plotting scripts (e.g. <code>plot_combo_lfw_vggface2_from_xls.py</code>) , data not read from .xls files (inserted as argument on command line) are stored as global variables in the scripts themselves.
In the evaluation scripts ("_eval ... .py_" scripts), passing --time the .txt output files will have date and time in their name.

### VGGFACE2

```bash
python3 eval_uncorrupted_vggface2_gender.py --gpu 0 --path ../trained
```
```bash
python3 conv_txt_to_xls.py --input results/vggface2/results.txt
```
```bash
python3 tabulate_vggface2_gender_from_xls.py --uncorrupted results/vggface2/results.xls
```

### LFW+
```bash
python3 eval_corrupted_lfw_gender.py --gpu 0 --path ../trained
```
```bash
python3 eval_corrupted_lfw_gender.py --gpu 0 --path ../trained --nocorruption
```
```bash
python3 conv_txt_to_xls.py --input results/lfw/corrupted_results.txt
```
```bash
python3 conv_txt_to_xls.py --input results/lfw/uncorrupted_results.txt
```
```bash
python3 plot_and_tabulate_lfw_from_xls.py --corrupted results/lfw/corrupted_results.xls --uncorrupted results/lfw/uncorrupted_results.xls
```
```bash
python3 plot_combo_lfw_vggface2_from_xls.py --corrupted results/lfw/corrupted_results.xls --uncorrupted results/lfw/uncorrupted_results.xls
```

### MIVIA
```bash
python3 eval_uncorrupted_mivia_gender.py --gpu 0 --path ../trained
```
```bash
python3 conv_txt_to_xls.py --input results/mivia/results.txt 
```
```bash
python3 tabulate_mivia_gender_from_xls.py --uncorrupted results/mivia/results.xls
```


### FERET
```bash
python3 eval_corrupted_feret_gender.py --gpu 0 --path ../trained
```
```bash
python3 eval_corrupted_feret_gender.py --gpu 0 --path ../trained --nocorruption
```
```bash
python3 conv_txt_to_xls.py --input results/feret/corrupted_results.txt
```
```bash
python3 conv_txt_to_xls.py --input results/feret/uncorrupted_results.txt
```
```bash
python3 plot_and_tabulate_feret_from_xls.py --corrupted results/feret/corrupted_results.xls --uncorrupted results/feret/uncorrupted_results.xls
```
```bash
python3 plot_combo_feret_lfw_vggface2_from_xls.py --corrupted results/feret/corrupted_results.xls --uncorrupted results/feret/uncorrupted_results.xls
```

## Project structure
The whole project should look like in this way:

```
gender
├── dataset
│   ├── cache
│   ├── data
│   ├── feret_aug_dataset.py
│   ├── face_models
│   ├── __pycache__
│   ├── mivia_dataset_gender.py
│   ├── feret_dataset_gender.py
│   ├── face_detector.py
│   ├── lfw_plus_aug_dataset.py
│   ├── vgg2_utils.py
│   ├── lfw_dataset_gender.py
│   └── vgg2_dataset_gender.py
├── trained
├── training
│   ├── corruptions.py
│   ├── __init__.py
│   ├── __pycache__
│   ├── keras-shufflenetV2
│   ├── keras-squeezenet
│   ├── cropout_test.py
│   ├── keras-squeeze-excite-network
│   ├── center_loss.py
│   ├── train.py
│   ├── model_build.py
│   ├── autoaug_test.py
│   ├── scratch_models
│   ├── dataset_tools.py
│   ├── keras-shufflenet
│   ├── keras_vggface
│   ├── DenseNet
│   ├── check_params.py
│   ├── autoaugment
│   └── ferplus_aug_dataset.py
├── evaluate
│   ├── plot_combo_feret_lfw_vggface2_from_xls.py
│   ├── conv_txt_to_xls.py
│   ├── eval_corrupted_feret_gender.py
│   ├── xls_models_tools.py
│   ├── plot_combo_lfw_vggface2_from_xls.py
│   ├── results
│   ├── tabulate_vggface2_gender_from_xls.py
│   ├── tabulate_mivia_gender_from_xls.py
│   ├── eval_corrupted_lfw_gender.py
│   ├── plot_and_tabulate_lfw_from_xls.py
│   ├── eval_uncorrupted_mivia_gender.py
│   ├── plot_and_tabulate_feret_from_xls.py
│   └── eval_uncorrupted_vggface2_gender.py
├── __init__.py
├── README.md
└── content.txt
```

## Acknowledgements
The code for generation of the corrupted dataset relies on the work from [github.com/hendrycks/robustness](github.com/hendrycks/robustness).

The code in this repository also includes open keras implementations of well-known CNN architectures:
* Shufflenet: https://github.com/arthurdouillard/keras-shufflenet
* ShufflenetV2: https://github.com/opconty/keras-shufflenetV2
* Squeezenet: https://github.com/rcmalli/keras-squeezenet
* SENet: https://github.com/titu1994/keras-squeeze-excite-network
* VGGFace: https://github.com/rcmalli/keras-vggface
* DenseNet: https://github.com/titu1994/DenseNet
* MobileNetV2: https://github.com/vvigilante/mobilenet_v2_keras
* VGG16: https://github.com/fchollet/deep-learning-models



