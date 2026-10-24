import torch
import numpy as np
import seaborn as sns
from scipy import io
import os

from utils.display import convert_to_color_, convert_from_color_


def GT_To_One_Hot(gt, class_count):
    '''
    Convet Gt to one-hot labels
    :param gt: 2D
    :param class_count:
    :return:
    '''
    GT_One_Hot = []  # 转化为one-hot形式的标签
    height=gt.shape[0]
    for i in range(height):

        temp = np.zeros(class_count, dtype=np.float32)
        # gai!!!!
        if gt[i] != -1:
            temp[int(gt[i]) - 1] = 1

        GT_One_Hot.append(temp)
    GT_One_Hot = np.reshape(GT_One_Hot, [height, class_count])
    return GT_One_Hot

def save_checkpoint(check_points, is_best, path) -> None:
    
    if is_best:
        # print("saving checkpoints")
        torch.save(check_points, path)
    
def compute_accuracy(accuracy, total, output, target):

    accuracy = accuracy
    total = total
    _, output = torch.max(output, dim=1) 

    # ignored_labels = data_loader.dataset.ignored_labels

    for out, pred in zip(output.view(-1), target.view(-1)):

        accuracy += (out.item() + 1) == pred.item()
        total += 1

    return accuracy , total

def convert_to_color(x, N_Classes, palette):
    
    if palette is None:
    # Generate color palette
        palette = {0: (0, 0, 0)}

    for k, color in enumerate(sns.color_palette("hls", N_Classes - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))

    invert_palette = {v: k for k, v in palette.items()}
    
    return convert_to_color_(x, palette=palette)

def generate_predict_map(accuracy, total, output, target, right_class_num, total_class_num, indices, predict_map):

    accuracy = accuracy
    total = total
    _, output = torch.max(output, dim=1) 

    # ignored_labels = data_loader.dataset.ignored_labels
    x_coor = indices[:, 0]
    y_coor = indices[:, 1]

    for out, pred, x, y in zip(output.view(-1), target.view(-1), x_coor.view(-1), y_coor.view(-1)):

        if out.item() + 1  == pred.item():
            accuracy = accuracy + 1
            right_class_num[pred.item()] = right_class_num[pred.item()] + 1

        total += 1
        total_class_num[pred.item()] = total_class_num[pred.item()] + 1
        predict_map[x, y] = out.item() + 1


    return accuracy , total, right_class_num, total_class_num, predict_map

def generate_feature_map(feature, x_coor, y_coor, img):

    m, n, _ = img.shape
    feature_map = torch.zeros(m, n, 256)

    for item, x, y in zip(feature, x_coor, y_coor):
        feature_map[x,y] = item

    return feature_map

def parameter_save(predict_map, args, feature_path):

    name = 'predict.mat'
    io.savemat(name, {'predict': predict_map})

    save_parameter = vars(args)
    torch.save(save_parameter, feature_path + '/args.pth')

def parameter_check(feature_path, args):

    check = False
    if os.path.exists(feature_path + '/args.pth'):
        previous_args = torch.load(feature_path + '/args.pth')
        check = (previous_args == vars(args))
    
    return check

