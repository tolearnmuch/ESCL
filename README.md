# Skilful Weather Nowcasting by Evolution-Similarity Contrastive Learning (ESCL)

This is the implementation of 'Skilful Weather Nowcasting by Evolution-Similarity Contrastive Learning' submitted for peer review.

![image](https://github.com/tolearnmuch/ESCL/blob/main/logdir/Figure%207.1.png)

## 1. Important updates

* 2022-10-26: This project is going to be released, please waiting.

* 2022-10-27: Upoad released models, inferencing codes, training codes and supplemental materials

* 2022-...

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

## 3. Inference with our trained ESCL on Shanghai-2020 & HKO-7

Released pre-trained model is available on our [mega drive](https://mega.nz/folder/lZ91CLhQ#yEpKYTeoj2RY9wAr9Kf-Fw).

* For do testing experiments on Shanghai-2020 & HKO-7, set parameters in configs.py and below is an example:

```
    mode = 'test' 
    model = 'ESCL'
    ini_mode = 'xavier'

    dataset_type = 'HKO'  
    dataset_root = r'D:\xyc\dataset\HKO-7'
    #dataset_root = r'D:\xyc\dataset\competition'
    #dataset_type = 'shanghai'

    in_len = 10
    out_len = 10

    img_width = 128
    img_height = 128
    
    pretrained_model = 'ESCL'

    use_gpu = True
    num_workers = 8
    device_ids = [0,1,2,3]
    device_ids_eval = [0]
    batch_size = 4
    test_batch_size = 1     # for save seqs, please set this one to be 1, and other cases could be like 4...
    train_max_epoch = 20
    learning_rate = 1e-4
    optim_betas = (0.9, 0.999)
    scheduler_gamma = 1.0

    train_print_fre = 100
    img_print_fre = 1000
    model_save_fre = 1
    log_dir = r'logdir'
    model_save_dir = r'D:\xyc\PrecipNowcastingBench\Benchmark_Precipitation_Nowcasting\BPN-master\save_models'
    test_imgs_save_dir = r'save_results'
```

then place pre-trained models .pth files in $model_save_dir$. and run

```
python main.py
```

## 4. Training your own models

For training the models on Shanghai-2020 and HKO-7 datasets, set the parameters as in configs.py, and here is an example:

```
    mode = 'train'
    model = 'ESCL'
    ini_mode = 'xavier'

    # dataset_type = 'HKO'
    # dataset_root = r'D:\xyc\dataset\HKO-7'
    dataset_root = r'D:\xyc\dataset\competition'
    dataset_type = 'shanghai'

    random_sampling = False
    random_iters = 10000

    in_len = 10
    out_len = 10

    img_width = 128
    img_height = 128
    #
    fine_tune = False
    pretrained_model = 'ESCL'

    use_gpu = True
    # num_workers = 8
    num_workers = 8
    device_ids = [0,1,2,3]
    device_ids_eval = [0]
    batch_size = 4
    test_batch_size = 1     # for save seqs, please set this one to be 1, and other cases could be like 4...
    train_max_epoch = 20
    learning_rate = 1e-4
    optim_betas = (0.9, 0.999)
    scheduler_gamma = 1.0

    train_print_fre = 100
    img_print_fre = 1000
    model_save_fre = 1
    log_dir = r'logdir'
    model_save_dir = r'D:\xyc\PrecipNowcastingBench\Benchmark_Precipitation_Nowcasting\BPN-master\save_models'
    test_imgs_save_dir = r'save_results'
```

then run

```
python main.py
```

## 6. More cases

More cases and details are available at our [supplemental materials](https://mega.nz/file/NQkiXaYR#IZeBbbeN4xhzTQBH7bTjgah4oIoQl71d4slYVhKaIsQ)

## 7. License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments



