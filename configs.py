class Configs(object):
    mode = 'train'      # train, test， valid
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
    # device_ids = [0]
    device_ids_eval = [0]
    # device_ids = [0]
    batch_size = 4
    test_batch_size = 1     # for save seqs, please set this one to be 1, and other cases could be like 4...
    train_max_epoch = 20
    learning_rate = 1e-4
    # learning_rate = 1e-5
    optim_betas = (0.9, 0.999)
    scheduler_gamma = 1.0

    train_print_fre = 100
    img_print_fre = 1000
    model_save_fre = 1
    log_dir = r'logdir'
    # model_save_dir = r'save_models'
    model_save_dir = r'D:\xyc\PrecipNowcastingBench\Benchmark_Precipitation_Nowcasting\BPN-master\save_models'
    test_imgs_save_dir = r'save_results'


configs = Configs()
# print('shanghai escl 2-10')




