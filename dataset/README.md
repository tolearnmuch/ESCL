# Data processing and loading scripts for Shanghai-2020 & HKO-7.

This is the implementation scripts of loading echo data from Shanghai-2020 and HKO-7 datasets.

## Dataset

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

## 2. Scripts of examples on creating dataset and dataloader

* For Shanghai-2020:

The main scripts traveling on Shanghai-2020 are in Shanghai_2020.py, with this file, 

For creating train dataloader,

```
train_dataset = dataset.Shanghai_2020(configs.dataset_root, seq_len=in_len + out_len, seq_interval=5,
                                              train=True,
                                              test=False, nonzero_points_threshold=6500)
train_dataloader = DataLoader(train_dataset, configs.batch_size, shuffle=True, num_workers=configs.num_workers,
                              drop_last=True)
```

For creating valid dataloader,

```
valid_dataset = dataset.Shanghai_2020(configs.dataset_root, seq_len=in_len + out_len, seq_interval=5,
                                              train=False,
                                              test=False, nonzero_points_threshold=None)
valid_dataloader = DataLoader(valid_dataset, configs.test_batch_size, shuffle=False,
                              num_workers=configs.num_workers)
```

For creating test dataloader,

```
test_dataset = dataset.Shanghai_2020(configs.dataset_root, seq_len=in_len + out_len, seq_interval=5,
                                     train=False,
                                     test=True, nonzero_points_threshold=None)
test_dataloader = DataLoader(test_dataset, configs.test_batch_size, shuffle=False,
                             num_workers=configs.num_workers)
```                                     

* For HKO-7:

For creating train dataloader,

```
train_dataset = dataset.HKO_7(configs.dataset_root, seq_len=in_len + out_len, seq_interval=5, train=True,
                                      test=False, nonzero_points_threshold=6500)
train_dataloader = DataLoader(train_dataset, configs.batch_size, shuffle=True, num_workers=configs.num_workers,
                              drop_last=True)
```

For creating valid dataloader,

```
valid_dataset = dataset.HKO_7(configs.dataset_root, seq_len=in_len + out_len, seq_interval=5, train=False,
                                      test=False, nonzero_points_threshold=None)
valid_dataloader = DataLoader(valid_dataset, configs.test_batch_size, shuffle=False,
                              num_workers=configs.num_workers)
```

For creating test dataloader,

```
test_dataset = dataset.HKO_7(configs.dataset_root, seq_len=in_len + out_len, seq_interval=5, train=False,
                             test=True, nonzero_points_threshold=None)
test_dataloader = DataLoader(test_dataset, configs.test_batch_size, shuffle=False,
                             num_workers=configs.num_workers)
```

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details





