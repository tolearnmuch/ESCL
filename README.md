# Skilful Weather Nowcasting by Evolution-Similarity Contrastive Learning (ESCL)

This is the implementation of 'Skilful Weather Nowcasting by Evolution-Similarity Contrastive Learning' submitted for peer review.

![image](https://github.com/tolearnmuch/ESCL/blob/main/logdir/Figure%207.1.png)

## 1. Important updates

* 2022-10-26: This project is going to be released, please waiting.

* ...

## 2. Getting started

The requirements of the hardware and software are given as below.

### 2.1. Prerequisites

> CPU: Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz 2.99 GHz (2 CPUs)
>
> GPU: GeForce GTX 3090 x 4
> 
> CUDA Version: 11.6
> 
> OS: Windows 10

### 2.2. Installing

Configure the virtual environment on Windows.

* Dependencies on Anaconda

```
python >= 3.6
pytorch >= 1.6
torchvision >= 0.7.0
tensorboard
imageio
tqdm
opencv-python
pillow
scikit-image
matplotlib
einops
```

### 2.3. Dataset

Datasets contain: [Shanghai-2020](https://doi.org/10.5281/zenodo.7251972) & [HKO-7](https://github.com/sxjscience/HKO-7)

For reproducing our experiments, the pre-precessing should be done at first.

* For Shanghai-2020:

```
python Shanghai_2020_preprocessing.py
```

The structure of Shanghai-2020 should be:

```
-test/
  -data/
  -examples/
-train/
  -data/
  -examples/
```

* For HKO-7:

```
python HKO_7_preprocessing.py
```

The structure of HKO-7 should be:

```
-radarPNG/
-radarPNG_mask/
-hko7_rainy_test_days.txt
-hko7_rainy_test_days_statistics.txt
-hko7_rainy_train_days.txt
-hko7_rainy_train_days_statistics.txt
-hko7_rainy_valid_days.txt
-hko7_rainy_valid_days_statistics.txt
```

* For examples on using dataset, please refer to https://github.com/tolearnmuch/ESCL/tree/main/dataset.

## Inference with our trained ESCL on Shanghai-2020 & HKO-7

Released pre-trained model is available on our [mega drive]()

## 4. Inference with released models

For downloading the [released models](https://mega.nz/folder/hA8mBKqA#WcSp7gl70OclmItphc7olA), the released models should be placed as:

>——./pretrained/pretrained_models/kth/ours_asvp
>
>——./pretrained/pretrained_models/bair_action_free/ours_asvp

and the pre-trained models for baseline should be placed as:

>——./pretrained/pretrained_models/kth/savp
>
>——./pretrained/pretrained_models/bair_action_free/asvp

### 4.1. Inference on KTH human action

* For running our released model, please run

```
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --input_dir data/kth --dataset_hparams sequence_length=30 --checkpoint pretrained_models/kth/ours_asvp/model-300000 --mode test --results_dir results_test_samples/kth --batch_size 3
```

* For running the baseline, please run

```
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --input_dir data/kth --dataset_hparams sequence_length=30 --checkpoint pretrained_models/kth/savp/model-300000 --mode test --results_dir results_test_samples/kth --batch_size 3
```

### 4.2. Inference on BAIR action-free robot pushing

* For running our released model, please run

```
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --input_dir data/bair --dataset_hparams sequence_length=22 --checkpoint logs/bair_action_free/ours_asvp/model-300000 --mode test --results_dir results_test_samples/bair_action_free --batch_size 8
```

* For running the baseline, please run

```
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --input_dir data/bair --dataset_hparams sequence_length=22 --checkpoint pretrained_models/bair_action_free/savp/model-300000 --mode test --results_dir results_test_samples/bair_action_free --batch_size 8
```

## 5. Training

* For training our model with active patterns and non-active patterns on KTH, please run

```
CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py --input_dir data/kth --dataset kth --model asvp --model_hparams_dict hparams/kth/ours_asvp/model_hparams.json --output_dir logs/kth/ours_asvp
```

* For training our model with active patterns and non-active patterns on BAIR action-free, please run

```
CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py --input_dir data/bair --dataset bair --model asvp --model_hparams_dict hparams/bair_action_free/ours_asvp/model_hparams.json --output_dir logs/bair_action_free/ours_asvp
```

## 6. More cases

Add additional notes about how to deploy this on a live system

## 7. License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments



