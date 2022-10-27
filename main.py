import os
import utils
import models
import dataset
import torch as t
from torch import nn
from configs import configs
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.loss_functions import weighted_l2_loss_radar
import sys

in_len = configs.in_len
out_len = configs.out_len

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


def ini_model_params(model, ini_mode='xavier'):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.Linear)):
            if ini_mode == 'xavier':
                nn.init.xavier_normal_(m.weight)
            elif ini_mode == 'orthogonal':
                nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


def out_interpolation20(output):
    new_out = []
    for i in range(19, 38):
        # interpolation = ((output[:, i] + output[:, i -1]) / 2).unsqueeze(1)
        # new_out.append(interpolation)
        new_out.append(output[:, i].unsqueeze(1))
    new_out = t.cat(new_out, dim=1)
    return new_out


def test(dBZ_threshold=10, if_valid=False, if_saveimg=True):
    if if_valid:
        print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  Start Valid")
    else:
        print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  Start Test")

    model_name = configs.model_save_dir + '/' + configs.pretrained_model + '.pth'
    #
    # """test"""
    if if_valid:
        pod, far, csi, bias, hss, ssim = valid_local(model_name, dBZ_threshold=10, dataset_type=configs.dataset_type,
                                                     save_seq=if_saveimg)
    else:
        pod, far, csi, bias, hss, ssim = test_local(model_name, dBZ_threshold=10, dataset_type=configs.dataset_type,
                                                    save_seq=if_saveimg)

    # print('Time: ' + datetime.now().strftime(
    #     '%Y-%m-%d %H:%M:%S') + '  Test:\tPOD: {:.4f}, FAR: {:.4f}, CSI: {:.4f}, BIAS: {:.4f}, HSS: {:.4f}, SSIM: {:.4f}'
    #       .format(t.mean(pod), t.mean(far), t.mean(csi), t.mean(bias), t.mean(hss), t.mean(ssim)))

    utils.save_test_results(configs.test_imgs_save_dir, pod, far, csi, bias, hss, ssim)


def test_local(model_name, dBZ_threshold=10, dataset_type='HKO', eval_by_seq=False, save_seq=False, eval_ssim=True):
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  Start test")

    # create dataloader
    if configs.dataset_type == 'HKO':
        valid_dataset = dataset.HKO_7(configs.dataset_root, seq_len=in_len + out_len, seq_interval=5, train=False,
                                      test=False, nonzero_points_threshold=None)
        valid_dataloader = DataLoader(valid_dataset, configs.test_batch_size, shuffle=False,
                                      num_workers=configs.num_workers)
        test_dataset = dataset.HKO_7(configs.dataset_root, seq_len=in_len + out_len, seq_interval=5, train=False,
                                     test=True, nonzero_points_threshold=None)
        test_dataloader = DataLoader(test_dataset, configs.test_batch_size, shuffle=False,
                                     num_workers=configs.num_workers)
    elif configs.dataset_type == 'shanghai':
        valid_dataset = dataset.Shanghai_2020(configs.dataset_root, seq_len=in_len + out_len, seq_interval=5,
                                              train=False,
                                              test=False, nonzero_points_threshold=None)
        valid_dataloader = DataLoader(valid_dataset, configs.test_batch_size, shuffle=False,
                                      num_workers=configs.num_workers)
        test_dataset = dataset.Shanghai_2020(configs.dataset_root, seq_len=in_len + out_len, seq_interval=5,
                                             train=False,
                                             test=True, nonzero_points_threshold=None)
        test_dataloader = DataLoader(test_dataset, configs.test_batch_size, shuffle=False,
                                     num_workers=configs.num_workers)
    elif configs.dataset_type == 'jiangsu':
        valid_dataset = dataset.Jiangsu_2022(configs.dataset_root, train=False)
        valid_dataloader = DataLoader(valid_dataset, configs.test_batch_size, shuffle=True,
                                      num_workers=configs.num_workers)
        valid_dataset = dataset.Jiangsu_2022(configs.dataset_root, train=False)
        test_dataloader = DataLoader(valid_dataset, 1, shuffle=False, num_workers=configs.num_workers)

    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
          'Load test dataset successfully...')

    # Model setting
    if configs.model == 'ESCL':
        model = models.ESCL(
            input_channels=1,
            output_sigmoid=True,
            # model architecture
            layers_per_block=(3, 3, 3, 3),
            hidden_channels=(32, 48, 48, 32),
            skip_stride=2,
            # convolutional tensor-train layers
            cell=r'convlstm',
            cell_params={
                "order": 3,
                "steps": 3,
                "ranks": 8},
            # convolutional parameters
            kernel_size=3).cuda()

    model.load_state_dict(t.load(model_name))
    if configs.use_gpu:
        device = t.device('cuda:0')
        if len(configs.device_ids_eval) > 1:
            model = nn.DataParallel(model, device_ids=configs.device_ids_eval, dim=0)
            model.to(device)
        else:
            model.to(device)
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  Load successfully {}".format(model_name))

    pod, far, csi, bias, hss, ssim = valid(model, test_dataloader, dBZ_threshold=dBZ_threshold,
                                           dataset=configs.dataset_type,
                                           eval_by_seq=True, save_seq=save_seq)

    print('Time: ' + datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S') + '  Test:\tPOD: {:.4f}, FAR: {:.4f}, CSI: {:.4f}, BIAS: {:.4f}, HSS: {:.4f}, SSIM: {:.4f}'
          .format(t.mean(pod), t.mean(far), t.mean(csi), t.mean(bias), t.mean(hss), t.mean(ssim)))

    # utils.save_test_results(configs.test_imgs_save_dir, pod, far, csi, bias, hss, ssim)
    return pod, far, csi, bias, hss, ssim


def valid_local(model_name, dBZ_threshold=10, dataset_type='HKO', eval_by_seq=False, save_seq=False, eval_ssim=True):
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  Start Eval")

    # create dataloader
    if configs.dataset_type == 'HKO':
        valid_dataset = dataset.HKO_7(configs.dataset_root, seq_len=in_len + out_len, seq_interval=5, train=False,
                                      test=False, nonzero_points_threshold=None)
        valid_dataloader = DataLoader(valid_dataset, configs.test_batch_size, shuffle=False,
                                      num_workers=configs.num_workers)
        test_dataset = dataset.HKO_7(configs.dataset_root, seq_len=in_len + out_len, seq_interval=5, train=False,
                                     test=True, nonzero_points_threshold=None)
        test_dataloader = DataLoader(test_dataset, configs.test_batch_size, shuffle=False,
                                     num_workers=configs.num_workers)
    elif configs.dataset_type == 'shanghai':
        valid_dataset = dataset.Shanghai_2020(configs.dataset_root, seq_len=in_len + out_len, seq_interval=5,
                                              train=False,
                                              test=False, nonzero_points_threshold=None)
        valid_dataloader = DataLoader(valid_dataset, configs.test_batch_size, shuffle=False,
                                      num_workers=configs.num_workers)
        test_dataset = dataset.Shanghai_2020(configs.dataset_root, seq_len=in_len + out_len, seq_interval=5,
                                             train=False,
                                             test=True, nonzero_points_threshold=None)
        test_dataloader = DataLoader(test_dataset, configs.test_batch_size, shuffle=False,
                                     num_workers=configs.num_workers)
    elif configs.dataset_type == 'jiangsu':
        valid_dataset = dataset.Jiangsu_2022(configs.dataset_root, train=False)
        valid_dataloader = DataLoader(valid_dataset, configs.test_batch_size, shuffle=True,
                                      num_workers=configs.num_workers)
        valid_dataset = dataset.Jiangsu_2022(configs.dataset_root, train=False)
        test_dataloader = DataLoader(valid_dataset, 1, shuffle=False, num_workers=configs.num_workers)

    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
          'Load valid dataset successfully...')

    if configs.model == 'ESCL':
        model = models.ESCL(
            input_channels=1,
            output_sigmoid=True,
            # model architecture
            layers_per_block=(3, 3, 3, 3),
            hidden_channels=(32, 48, 48, 32),
            skip_stride=2,
            # convolutional tensor-train layers
            cell=r'convlstm',
            cell_params={
                "order": 3,
                "steps": 3,
                "ranks": 8},
            # convolutional parameters
            kernel_size=3).cuda()

    model.load_state_dict(t.load(model_name))
    if configs.use_gpu:
        device = t.device('cuda:0')
        if len(configs.device_ids_eval) > 1:
            model = nn.DataParallel(model, device_ids=configs.device_ids_eval, dim=0)
            model.to(device)
        else:
            model.to(device)
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  Load successfully {}".format(model_name))

    pod, far, csi, bias, hss, ssim = valid(model, valid_dataloader, dBZ_threshold=dBZ_threshold,
                                           dataset=configs.dataset_type,
                                           eval_by_seq=True, save_seq=False)

    print('Time: ' + datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S') + '  Test:\tPOD: {:.4f}, FAR: {:.4f}, CSI: {:.4f}, BIAS: {:.4f}, HSS: {:.4f}, SSIM: {:.4f}'
          .format(t.mean(pod), t.mean(far), t.mean(csi), t.mean(bias), t.mean(hss), t.mean(ssim)))

    # utils.save_test_results(configs.test_imgs_save_dir, pod, far, csi, bias, hss, ssim)
    return pod, far, csi, bias, hss, ssim


def valid(model, dataloader, dBZ_threshold=10, dataset='HKO', eval_by_seq=False, save_seq=False, eval_ssim=True):
    model.eval()
    pod = []
    far = []
    csi = []
    bias = []
    hss = []
    index = []
    ssim = []
    score = []

    # # 1 card to 1 device for some problems when like convlstm cannot work on 4 gpus
    # device = t.device('cuda:0')
    # model = nn.DataParallel(model, device_ids=configs.device_ids_eval, dim=0)
    # model.to(device)

    with t.no_grad():
        # Address data from dataloader
        # bar
        l_t = len(dataloader.dataset)
        fq = round(l_t / 100)
        i = 0
        for iter, data in enumerate(dataloader):
            if iter % fq == 0:
                i += 1
            # print(iter%fq, l_t)
            print("\r", end="")
            print("valid progress: {}%: ".format(i), "▋" * (i // 2), end="")
            # if iter >= 2:
            #     break
            input = data[:, 0:in_len]
            ground_truth = data[:, in_len:(in_len + out_len)]
            if configs.use_gpu:
                device = t.device('cuda:0')
                input = input.to(device)
                ground_truth = ground_truth.to(device)
            if configs.model == 'ESCL':
                output, output_f1, output_f2 = model(inputs=input,
                                                     input_frames=in_len,
                                                     future_frames=out_len,
                                                     output_frames=in_len + out_len - 1,
                                                     teacher_forcing=False,
                                                     scheduled_sampling_ratio=0)
                ground_truth = t.cat([input[:, 1:in_len], ground_truth], dim=1)

            pod_, far_, csi_, bias_, hss_, index_ = utils.crosstab_evaluate(output, ground_truth, dBZ_threshold, 70,
                                                                            dataset)
            pod.append(pod_.data)
            far.append(far_.data)
            csi.append(csi_.data)
            bias.append(bias_.data)
            hss.append(hss_.data)
            index.append(index_)
            if eval_ssim:
                ssim_ = utils.compute_ssim(output.cpu().numpy(), ground_truth.cpu().numpy())
                ssim.append(ssim_)
            # print(index_.size())

            # save seqs ?
            if save_seq:
                utils.save_test_imgs(configs.test_imgs_save_dir, iter, input, ground_truth, output,
                                     configs.dataset_type, save_mode='simple')
            # break
        # index = t.cat(index, dim=1)  # not appropriate for ours
        index = t.cat(index, dim=0)
        data_num = index.numel()
        # print(data_num)
        # the ground-truth sample which has no rainfall preddiction hits will not be included in calculation
        # cal_num = index.size()[1] - t.sum(index, dim=1) if eval_by_seq is True else data_num - t.sum(index)  # not apppri...
        # print(cal_num)
        cal_num = index.size()[0] - t.sum(index, dim=0) if eval_by_seq is True else data_num - t.sum(index)
        # print(cal_num)
        # not app....
        # pod = t.sum(t.cat(pod, dim=1), dim=1) / cal_num if eval_by_seq is True else t.sum(t.cat(pod, dim=1)) / cal_num
        # far = t.sum(t.cat(far, dim=1), dim=1) / cal_num if eval_by_seq is True else t.sum(t.cat(far, dim=1)) / cal_num
        # csi = t.sum(t.cat(csi, dim=1), dim=1) / cal_num if eval_by_seq is True else t.sum(t.cat(csi, dim=1)) / cal_num
        # bias = t.sum(t.cat(bias, dim=1), dim=1) / cal_num if eval_by_seq is True else t.sum(
        #     t.cat(bias, dim=1)) / cal_num
        # hss = t.sum(t.cat(hss, dim=1), dim=1) / cal_num if eval_by_seq is True else t.sum(t.cat(hss, dim=1)) / cal_num
        # if eval_ssim:
        #     ssim = t.mean(t.cat(ssim, dim=1), dim=1) if eval_by_seq is True else t.mean(t.cat(ssim, dim=1))
        # else:
        #     ssim = None
        pod = t.sum(t.cat(pod, dim=0), dim=0) / cal_num if eval_by_seq is True else t.sum(t.cat(pod, dim=0)) / cal_num
        far = t.sum(t.cat(far, dim=0), dim=0) / cal_num if eval_by_seq is True else t.sum(t.cat(far, dim=0)) / cal_num
        csi = t.sum(t.cat(csi, dim=0), dim=0) / cal_num if eval_by_seq is True else t.sum(t.cat(csi, dim=0)) / cal_num
        bias = t.sum(t.cat(bias, dim=0), dim=0) / cal_num if eval_by_seq is True else t.sum(
            t.cat(bias, dim=0)) / cal_num
        hss = t.sum(t.cat(hss, dim=0), dim=0) / cal_num if eval_by_seq is True else t.sum(t.cat(hss, dim=0)) / cal_num
        if eval_ssim:
            ssim = t.mean(t.cat(ssim, dim=0), dim=0) if eval_by_seq is True else t.mean(t.cat(ssim, dim=0))
        else:
            ssim = None
    model.train()

    # # 1 card to 4 cards
    # device = t.device('cuda:0')
    # model = nn.DataParallel(model, device_ids=configs.device_ids, dim=0)
    # model.to(device)

    return pod, far, csi, bias, hss, ssim


def train():
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Train mode is go. ")

    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' Model structure: \t {}'.format(configs.model))
    # Model setting
    if configs.model == 'ESCL':
        model = models.ESCL(
            input_channels=1,
            output_sigmoid=True,
            # model architecture
            layers_per_block=(3, 3, 3, 3),
            hidden_channels=(32, 48, 48, 32),
            skip_stride=2,
            # convolutional tensor-train layers
            cell=r'convlstm',
            cell_params={
                "order": 3,
                "steps": 3,
                "ranks": 8},
            # convolutional parameters
            kernel_size=3).cuda()

    # Pre-training setting
    if configs.fine_tune:
        print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
              'Fine tuning based on pre-trained model: {}.'.format(configs.pretrained_model))
        # contune training
        model.load_state_dict(t.load(configs.model_save_dir + '/' + configs.pretrained_model + '.pth'))
    else:
        print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
              'Learning from scratch. Initializing the model params...')
        ini_model_params(model, configs.ini_mode)

    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  Load Model Successfully")

    # GPU setting
    if configs.use_gpu:
        print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' Using GPU... ids: \t{}'.format(
            str(configs.device_ids)))
        device = t.device('cuda:0')
        if len(configs.device_ids) > 1:
            model = nn.DataParallel(model, device_ids=configs.device_ids, dim=0)
            model.to(device)
            # model.cuda()
        else:
            model.to(device)
    else:
        print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + 'Using CPU...')

    # Create dataloader
    if configs.dataset_type == 'HKO':
        train_dataset = dataset.HKO_7(configs.dataset_root, seq_len=in_len + out_len, seq_interval=5, train=True,
                                      test=False, nonzero_points_threshold=6500)
        train_dataloader = DataLoader(train_dataset, configs.batch_size, shuffle=True, num_workers=configs.num_workers,
                                      drop_last=True)
        valid_dataset = dataset.HKO_7(configs.dataset_root, seq_len=in_len + out_len, seq_interval=5, train=False,
                                      test=False, nonzero_points_threshold=None)
        valid_dataloader = DataLoader(valid_dataset, configs.test_batch_size, shuffle=False,
                                      num_workers=configs.num_workers)
    elif configs.dataset_type == 'shanghai':
        train_dataset = dataset.Shanghai_2020(configs.dataset_root, seq_len=in_len + out_len, seq_interval=5,
                                              train=True,
                                              test=False, nonzero_points_threshold=6500)
        train_dataloader = DataLoader(train_dataset, configs.batch_size, shuffle=True, num_workers=configs.num_workers,
                                      drop_last=True)
        valid_dataset = dataset.Shanghai_2020(configs.dataset_root, seq_len=in_len + out_len, seq_interval=5,
                                              train=False,
                                              test=False, nonzero_points_threshold=None)
        valid_dataloader = DataLoader(valid_dataset, configs.test_batch_size, shuffle=False,
                                      num_workers=configs.num_workers, drop_last=True)
    elif configs.dataset_type == 'jiangsu':
        train_dataset = dataset.Jiangsu_2022(configs.dataset_root, train=True)
        train_dataloader = DataLoader(train_dataset, configs.batch_size, shuffle=True, num_workers=configs.num_workers)
        valid_dataset = dataset.Jiangsu_2022(configs.dataset_root, train=False)
        valid_dataloader = DataLoader(valid_dataset, configs.test_batch_size, shuffle=True,
                                      num_workers=configs.num_workers)
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
          'Load dataset successfully...')

    # Definition of loss function
    criterion1 = nn.MSELoss()
    lam1 = 1.0
    criterion2 = nn.L1Loss()
    lam2 = 1.0

    # Other references
    optimizer = t.optim.Adam(model.parameters(), lr=configs.learning_rate, betas=configs.optim_betas)
    scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=configs.scheduler_gamma)

    writer = SummaryWriter(log_dir=configs.log_dir)
    valid_log_path = os.path.join(configs.log_dir, 'valid_record.txt')
    valid_log = open(valid_log_path, 'w')

    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  Start Train")
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  Start Train", file=valid_log)
    train_global_step = 0
    max_score = 0

    """Training & Validation in epochs"""
    for epoch in range(configs.train_max_epoch):
        # epoch += 15
        """Training"""
        for iter, data in enumerate(train_dataloader):
            # randomly sampling
            if configs.random_sampling:
                if iter * configs.batch_size >= configs.random_iters:
                    break

            # Address data from dataloader
            input = data[:, 0:in_len]
            ground_truth = data[:, in_len:(in_len + out_len)]
            if configs.use_gpu:
                input = input.to(device)
                ground_truth = ground_truth.to(device)
            # for teacher forcing true
            # input = t.cat([input, ground_truth], dim=1)
            optimizer.zero_grad()

            # Prepare output (generating output on input) and ground-truth
            if configs.model == 'ESCL':
                output, output_f1, output_f2 = model(inputs=input,
                                                     input_frames=in_len,
                                                     future_frames=out_len,
                                                     output_frames=in_len + out_len - 1,
                                                     teacher_forcing=False,
                                                     scheduled_sampling_ratio=(epoch / configs.train_max_epoch))
                ground_truth = t.cat([input[:, 1:in_len], ground_truth], dim=1)

            # Calculate loss and backward
            # loss1 = weighted_l2_loss_radar(output, ground_truth)
            # loss2 = vgg_loss(output, ground_truth)
            loss1 = criterion1(output, ground_truth)
            loss2 = criterion2(output, ground_truth)
            if configs.model == 'ESCL':
                loss3 = criterion2(output_f1, output_f2)
                loss = lam1 * loss1 + lam2 * loss2 + lam2 * loss3
            else:
                loss = lam1 * loss1 + lam2 * loss2
            loss.backward()
            optimizer.step()
            train_global_step += 1

            # Print loss & log loss
            if (iter + 1) % configs.train_print_fre == 0:
                print('Time: ' + datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S') + '  Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format((epoch + 1), (
                        iter + 1) * configs.batch_size, len(train_dataset), 100. * (iter + 1) / len(
                    train_dataloader), loss.item()))
                writer.add_scalar('Train/Loss/loss', loss.item(), train_global_step)
                # writer.add_scalar('Train/Loss/loss/loss1', loss1.item(), train_global_step)
                # writer.add_scalar('Train/Loss/loss/loss2', loss2.item(), train_global_step)

            # if (iter + 1) % configs.img_print_fre == 0:
            #     utils.img_seq_summary(ground_truth, train_global_step, 'Train/Radar/Ground_truth', writer)
            #     utils.img_seq_summary(output, train_global_step, 'Train/Radar/Prediction', writer)
            # break

        # Save by epochs
        if (epoch + 1) % configs.model_save_fre == 0:
            if len(configs.device_ids) > 1:
                t.save(model.module.state_dict(),
                       configs.model_save_dir + '/' + configs.model + '_epoch' + str(epoch + 1) + '.pth')
            else:
                t.save(model.state_dict(),
                       configs.model_save_dir + '/' + configs.model + '_epoch' + str(epoch + 1) + '.pth')

        model_name = configs.model_save_dir + '/' + configs.model + '_epoch' + str(epoch + 1) + '.pth'

        """Validate"""
        pod, far, csi, bias, hss, ssim = valid_local(model_name, dBZ_threshold=10, dataset_type=configs.dataset_type)

        """Validate"""
        # pod, far, csi, bias, hss, ssim = valid(model, valid_dataloader, dBZ_threshold=10, dataset=configs.dataset_type)
        # print('Time: ' + datetime.now().strftime(
        #     '%Y-%m-%d %H:%M:%S') + '  Train Epoch: {}\tPOD: {:.4f}, FAR: {:.4f}, CSI: {:.4f}, BIAS: {:.4f}, HSS: {:.4f}, SSIM: {:.4f}'.format(
        #     (epoch + 1), pod, far, csi, bias, hss, ssim))
        # print('Time: ' + datetime.now().strftime(
        #     '%Y-%m-%d %H:%M:%S') + '  Train Epoch: {}\tPOD: {:.4f}, FAR: {:.4f}, CSI: {:.4f}, BIAS: {:.4f}, HSS: {:.4f}, SSIM: {:.4f}'.format(
        #     (epoch + 1), pod, far, csi, bias, hss, ssim), file=valid_log)
        # writer.add_scalar('Valid/pod', pod, epoch + 1)
        # writer.add_scalar('Valid/far', far, epoch + 1)
        # writer.add_scalar('Valid/csi', csi, epoch + 1)
        # writer.add_scalar('Valid/bias', bias, epoch + 1)
        # writer.add_scalar('Valid/hss', hss, epoch + 1)
        # writer.add_scalar('Valid/ssim', ssim, epoch + 1)

        scheduler.step()

        csi = t.mean(csi)
        """Save model"""
        # Save best model
        if csi > max_score:
            max_score = csi
            print('Time: ' + datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S') + '  Train Epoch: {}\t  Save the current best model'.format(epoch + 1))
            print('Time: ' + datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S') + '  Train Epoch: {}\t  Save the current best model'.format(epoch + 1),
                  file=valid_log)
            if len(configs.device_ids) > 1:
                t.save(model.module.state_dict(), configs.model_save_dir + '/' + configs.model + '.pth')
            else:
                t.save(model.state_dict(), configs.model_save_dir + '/' + configs.model + '.pth')
        # # Save by epochs
        # if (epoch + 1) % configs.model_save_fre == 0:
        #     if len(configs.device_ids) > 1:
        #         t.save(model.module.state_dict(),
        #                configs.model_save_dir + '/' + configs.model + '_epoch' + str(epoch + 1) + '.pth')
        #     else:
        #         t.save(model.state_dict(),
        #                configs.model_save_dir + '/' + configs.model + '_epoch' + str(epoch + 1) + '.pth')

    writer.close()


def main():
    if configs.mode == 'train':
        train()
    elif configs.mode == 'test':
        test(if_saveimg=False)
    elif configs.mode == 'valid':
        test(if_valid=True, if_saveimg=False)


if __name__ == '__main__':
    main()
