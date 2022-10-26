# Skilful Weather Nowcasting by Evolution-Similarity Contrastive Learning (ESCL)

This is the implementation of 'Skilful Weather Nowcasting by Evolution-Similarity Contrastive Learning' submitted for peer review.

![image](https://github.com/tolearnmuch/ESCL/blob/main/logdir/Figure%201.3.png)

## 1. Important updates

* 2022-10-26: This project is going to be released, please waiting.

* ...

## 2. Getting started

The requirements of the hardware and software are given as below.

### 2.1. Prerequisites

> CPU: Intel(R) Core(TM) i7-6900K CPU @ 3.20GHz
>
> GPU: GeForce GTX 1080 Ti
> 
> CUDA Version: 10.2
> 
> OS: Ubuntu 16.04.6 LTS

### 2.2. Installing

Configure the virtual environment on Ubuntu.

* Create a virtual with python 3.6

```
conda create -n asvp python=3.6
conda activate asvp
```

* Install requirements (Please pay attention that we use tensorflow-gpu==1.10.0)

```
pip install -r requirements.txt
```

* Additionally install ffmpeg

```
conda install x264 ffmpeg -c conda-forge
```

Here the virtual env is created on Ubuntu.

### 2.3. Dataset

Datasets contain: [KTH human action dataset](https://www.csc.kth.se/cvap/actions/) & [BAIR action-free robot pushing dataset](https://sites.google.com/view/sna-visual-mpc/). For reproducing the experiment, the processed dataset should be downloaded:

* For KTH, raw data and subsequence file should be downloaded firstly. In this turn of submission, please temporarily download from:

[raw data](https://mega.nz/folder/JREhlAKB#U26ufSZcVSiw0EOOlW6pMw) and [subsequence file](https://mega.nz/folder/EVMiRJhB#Gboh1r5PmbqGv97db2974w). After downloading, drag all .zip and .tar.gz files into ./data directory, and run

```
bash data/preprocess_kth.sh
```

Then all preprocessed and subsequence splitted frames are obtained in ./data/kth/processed.

* If you only need to inference with released models, please run the code below for converting images into tfrecords for inference

```
bash data/kth2tfrecords.sh 
```

else, please skip this step and turn to Part 3 for separating active patterns along with non-active ones from videos.

## 3. Active pattern mining

Active pattern mining is necessary only for training and there is no need to do this if only with respective to inference with released model.

* When trying to separate active patterns along with non-active ones from videos, please refer to [details](https://github.com/Anonymous-Submission-ID/Anonymous-Submission/tree/main/separating_active_patterns/).

* After all active patters and non-active patterns are mined, these images are convertted to tfrecords for training.

```
bash data/kth2tfrecords_ap.sh
```

The final data could be downloaded from [drive](https://mega.nz/folder/VVlUiZII#kqCMjIRfCoS4IoOuMjTXZg/).

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



