import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from imageio import imwrite
from skimage.transform import resize
from PIL import Image
# import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import xlwt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# from configs import configs


def img_seq_summary(img_seq, global_step, name_scope, writer):
    seq_len = img_seq.size()[0]
    for i in range(seq_len):
        writer.add_images(name_scope + '/Img' + str(i + 1), img_seq[i], global_step)


def save_test_results(log_dir, pod, far, csi, bias, hss, ssim=None):
    test_results_path = os.path.join(log_dir, 'test_results.xls')
    work_book = xlwt.Workbook(encoding='utf-8')
    sheet = work_book.add_sheet('sheet')
    sheet.write(0, 0, 'pod')
    for col, label in enumerate(pod.tolist()):
        sheet.write(0, 1 + col, str(label))
    sheet.write(1, 0, 'far')
    for col, label in enumerate(far.tolist()):
        sheet.write(1, 1 + col, str(label))
    sheet.write(2, 0, 'csi')
    for col, label in enumerate(csi.tolist()):
        sheet.write(2, 1 + col, str(label))
    sheet.write(3, 0, 'bias')
    for col, label in enumerate(bias.tolist()):
        sheet.write(3, 1 + col, str(label))
    sheet.write(4, 0, 'hss')
    for col, label in enumerate(hss.tolist()):
        sheet.write(4, 1 + col, str(label))
    if ssim is not None:
        sheet.write(5, 0, 'ssim')
        for col, label in enumerate(ssim.tolist()):
            sheet.write(5, 1 + col, str(label))
    work_book.save(test_results_path)


def save_test_imgs(log_dir, index, input, ground_truth, output, dataset='HKO_7', save_mode='integral'):
    if dataset == 'HKO':
        input = 70.0 * input - 10.0
        output = 70.0 * output - 10.0
        ground_truth = 70.0 * ground_truth - 10.0
        # input = 70.0 * input
        # output = 70.0 * output
        # ground_truth = 70.0 * ground_truth
    if dataset == 'shanghai':
        input = 70.0 * input
        output = 70.0 * output
        ground_truth = 70.0 * ground_truth
    # input_seq_len = input.size()[0]
    # out_seq_len = output.size()[0]
    # not app..
    input_seq_len = input.size()[1]
    out_seq_len = output.size()[1]
    height = input.size()[4]
    width = input.size()[3]
    x = np.arange(0, width)
    y = np.arange(height, 0, -1)
    levels = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
    cmp = mpl.colors.ListedColormap(['white', 'lightskyblue', 'cyan', 'lightgreen', 'limegreen', 'green',
                                     'yellow', 'orange', 'chocolate', 'red', 'firebrick', 'darkred', 'fuchsia',
                                     'purple'], 'indexed')
    if not os.path.exists(os.path.join(log_dir, 'sample' + str(index + 1))):
        os.makedirs(os.path.join(log_dir, 'sample' + str(index + 1)))
    for i in range(input_seq_len):
        # x = rearrange(x, 'b l c h w -> (b l) c h w')   # set batchsize = 1, and adapt the order or [:, i]{batchsize=1, seqlen}
        img = input[:, i].squeeze().cpu().numpy()

        plt.contourf(x, y, img, levels=levels, extend='both', cmap=cmp)
        save_fig_path = os.path.join(log_dir, 'sample' + str(index + 1), 'input' + str(i + 1))
        if save_mode == 'simple':
            plt.xticks([])
            plt.yticks([])
            plt.savefig(save_fig_path, bbox_inches='tight', dpi=600)
        elif save_mode == 'integral':
            plt.title('Input')
            plt.xlabel('Timestep' + str(i + 1))
            plt.colorbar()
            plt.savefig(save_fig_path, dpi=600)
        plt.clf()
    for i in range(out_seq_len):
        img = output[:, i].squeeze().cpu().numpy()  # use [:, i] to replace [i], for the special seq: [bs, l, c, w, h]
        plt.contourf(x, y, img, levels=levels, extend='both', cmap=cmp)
        save_fig_path = os.path.join(log_dir, 'sample' + str(index + 1), 'output' + str(i + 1))
        if save_mode == 'simple':
            plt.xticks([])
            plt.yticks([])
            plt.savefig(save_fig_path, bbox_inches='tight', dpi=600)
        elif save_mode == 'integral':
            plt.title('Output')
            plt.xlabel('Timestep' + str(i + 1))
            plt.colorbar()
            plt.savefig(save_fig_path, dpi=600)
        plt.clf()
    for i in range(out_seq_len):
        img = ground_truth[:, i].squeeze().cpu().numpy()   # the same as above
        plt.contourf(x, y, img, levels=levels, extend='both', cmap=cmp)
        save_fig_path = os.path.join(log_dir, 'sample' + str(index + 1), 'ground_truth' + str(i + 1))
        if save_mode == 'simple':
            plt.xticks([])
            plt.yticks([])
            plt.savefig(save_fig_path, bbox_inches='tight', dpi=600)
        elif save_mode == 'integral':
            plt.title('Ground_truth')
            plt.xlabel('Timestep' + str(i + 1))
            plt.colorbar()
            plt.savefig(save_fig_path, dpi=600)
        plt.clf()
    return

# def img_seq_summary(img_seq, global_step, name_scope, writer):
#     batch_size = img_seq.size()[0]
#     for i in range(batch_size):
#         writer.add_images(name_scope + '/Seq' + str(i + 1), img_seq[i], global_step)
#
#
# def generate_index(i):
#     if i < 10:
#         i = '00' + str(i)
#     elif 10 <= i < 100:
#         i = '0' + str(i)
#     else:
#         i = str(i)
#     return i
#
#
# def save_test_imgs(output, save_dir, index):
#     case_index = generate_index(index)
#     seq_len = output.size()[1]
#     seq = output[0, :, 0, :, :]
#     for i in range(seq_len):
#         seq_index = generate_index(i + 1)
#         radar_img = (np.array(seq[i]) * 255).astype(np.uint8)
#         radar_img = np.array(Image.fromarray(radar_img).resize((560, 480)))  # 560, 480 !
#         radar_img_save_folder = os.path.join(save_dir, 'Radar', case_index)
#         if not os.path.exists(radar_img_save_folder):
#             os.makedirs(radar_img_save_folder)
#         imwrite(os.path.join(radar_img_save_folder, 'radar' + '_' + seq_index + '.png'), radar_img)
#
# """"""
#
# def save_test_imgs_windcal(output, save_dir, index):
#     case_index = generate_index(index)
#     seq_len = output.size()[1]
#     seq = output[0, :, 0, :, :]
#     max_img = (np.array(seq[0]) * 255).astype(np.uint8)
#     for i in range(seq_len):
#         seq_index = generate_index(i + 1)
#         radar_img = (np.array(seq[i]) * 255).astype(np.uint8)
#
#         # radar_img = np.array(Image.fromarray(radar_img).resize((560, 480)))  # 560, 480 !
#         # avarage the high response  doing fusion
#         for pos_i in range(480):
#             for pos_j in range(560):
#                 if radar_img[pos_i, pos_j] > max_img[pos_i, pos_j]:
#                     max_img[pos_i, pos_j] = radar_img[pos_i, pos_j]
#
#     for i in range(seq_len):
#         if i == 9 or i == 14 or i == 19:
#             if i == 9:
#                 max_img = (np.array(seq[18]) * 255).astype(np.uint8)
#             if i == 14:
#                 max_img = (np.array(seq[17]) * 255).astype(np.uint8)
#             if i == 19:
#                 max_img = (np.array(seq[16]) * 255).astype(np.uint8)
#         seq_index = generate_index(i + 1)
#         radar_img_save_folder = os.path.join(save_dir, 'Wind', case_index)
#         if not os.path.exists(radar_img_save_folder):
#             os.makedirs(radar_img_save_folder)
#
#         # fuse high response area, just
#
#         imwrite(os.path.join(radar_img_save_folder, 'wind' + '_' + seq_index + '.png'), max_img)
#
#
# def fuse_save_test_imgs(output, output_hq, save_dir, index):
#     case_index = generate_index(index)
#     seq_len = output.size()[1]
#     seq = output[0, :, 0, :, :]
#     seq_hq = output_hq[0, :, 0, :, :]
#     for i in range(seq_len):
#         seq_index = generate_index(i + 1)
#         radar_img = (np.array(seq[i]) * 255).astype(np.uint8)
#         radar_img_hq = (np.array(seq_hq[i]) * 255).astype(np.uint8)
#
#         radar_img = np.array(Image.fromarray(radar_img).resize((560, 480)))  # 560, 480 !
#         # avarage the high response  doing fusion
#         thres_highres = np.floor(30.0 / 70.0 * 255.0)
#         for pos_i in range(560):
#             for pos_j in range(480):
#                 if radar_img[0, pos_i, pos_j] > thres_highres and radar_img_hq[0, pos_i, pos_j] > thres_highres:
#                     fuse_v = (radar_img[0, pos_i, pos_j] + radar_img_hq[0, pos_i, pos_j]) / 2
#                     radar_img_hq[0, pos_i, pos_j] = np.floor(fuse_v)
#                 if radar_img[0, pos_i, pos_j] > thres_highres >= radar_img_hq[0, pos_i, pos_j]:
#                     fuse_v = radar_img[0, pos_i, pos_j]
#                     radar_img_hq[0, pos_i, pos_j] = np.floor(fuse_v)
#                 if radar_img[0, pos_i, pos_j] <= thres_highres < radar_img_hq[0, pos_i, pos_j]:
#                     radar_img_hq[0, pos_i, pos_j] = radar_img_hq[0, pos_i, pos_j]
#
#
#         radar_img_save_folder = os.path.join(save_dir, 'Radar', case_index)
#         if not os.path.exists(radar_img_save_folder):
#             os.makedirs(radar_img_save_folder)
#
#         # fuse high response area, just
#
#         imwrite(os.path.join(radar_img_save_folder, 'radar' + '_' + seq_index + '.png'), radar_img_hq)
