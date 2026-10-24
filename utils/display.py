from turtle import position
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import copy
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns
import logging, time
import os


def get_timestamp():
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%Y%m%d")
    return timestampDate + "-" + timestampTime


def setup_logger(logger_name, root, level=logging.INFO, screen=False, tofile=False):
    """set up logger"""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, "{}_{}.log".format(logger_name, get_timestamp()))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)
    return lg

def generate_t_sne(predict_mask, masks, coor):

    for index, (mask, position) in enumerate(zip(masks, coor)):

        predict_mask[position[0], position[1], :] = mask

    return predict_mask


def t_sne(HSI_or_high_feature, label,save_path, show_figure=False):

    """
    :param HSI_or_high_feature: input data (HSI or high feature)---> (H,W,B)
    :param label:  input label ---> (H,W)
    :param save_path:  save figure path
    :return: figure
    """

    print(HSI_or_high_feature.shape)

    HSI_or_high_feature = HSI_or_high_feature.transpose(2, 0, 1)
    print(HSI_or_high_feature.shape)

    d = np.zeros_like(label)
    d[label > 0] = True

    mask = copy.deepcopy(d)
    mask = mask.astype(np.bool)

    out = []
    for i in range(HSI_or_high_feature.shape[0]):

        img = HSI_or_high_feature[i, :, :]
        img = img.astype(np.float32)

        b = img[mask]
        b = b.reshape(1, -1)

        if i == 0:
            out = b
            continue
        out = np.concatenate((out, b), 0)

    HSI_or_high_feature = out.transpose(1, 0)
    print(HSI_or_high_feature.shape)
    label = label[mask]

    labels_num = np.max(label)
    area = np.pi * 1.5 ** 2  # 点面积

    HSI_or_high_feature_reduced = TSNE(n_components=2, init="pca", random_state=0).fit_transform(HSI_or_high_feature)

    for i in range(labels_num):
        print(i)
        coo_x = np.where(label == (i + 1))

        color_one = cm.rainbow(int(255 / labels_num * (i + 1)))  # 为了使得颜色有区分度，把0-255颜色区间分为9分,然后把标签映射到一个区间
        color_ = np.array(color_one).reshape(1, -1)

        plt.scatter(HSI_or_high_feature_reduced[coo_x, 0], HSI_or_high_feature_reduced[coo_x, 1], marker='o', s=area, c=color_)

    plt.box(False)

# 去掉X轴和Y轴的刻度
    plt.xticks([])
    plt.yticks([])
    plt.savefig(save_path, dpi=400)

    if not show_figure:

        plt.show()

    print("Visualization is complete")


def metrics(prediction, target, n_classes=None):
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).

    Args:
        prediction: list of predicted labels
        target: list of target labels
        n_classes (optional): number of classes, max(target) by default
    Returns:
        accuracy, accuracy by class, confusion matrix
    """
    
    ignored_mask = np.zeros(target.shape[:2], dtype=np.bool)
    ignored_mask[target < 0] = True
    ignored_mask = ~ignored_mask
    target = target[ignored_mask]
    prediction = prediction[ignored_mask]
    results = {}

    n_classes = np.max(target) + 1 if n_classes is None else n_classes

    cm = confusion_matrix(
        target,
        prediction,
        labels=range(1, n_classes+1))

    results["Confusion matrix"] = cm

    # Compute global accuracy
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy /= float(total)

    results["Accuracy"] = accuracy * 100.0

    # Compute accuracy of each class
    class_acc = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            acc = cm[i, i] / np.sum(cm[i, :])
        except ZeroDivisionError:
            acc = 0.
        class_acc[i] = acc

    results["class acc"] = class_acc * 100.0
    results['AA'] = np.mean(class_acc) * 100.0
    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
         float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa * 100.0

    return results

def show_results(results, label_values=None, agregated=False):
    text = ""

    if agregated:
        accuracies = [r["Accuracy"] for r in results]
        aa = [r['AA'] for r in results]
        kappas = [r["Kappa"] for r in results]
        class_acc = [r["class acc"] for r in results]

        class_acc_mean = np.mean(class_acc, axis=0)
        class_acc_std = np.std(class_acc, axis=0)
        cm = np.mean([r["Confusion matrix"] for r in results], axis=0)
        text += "Agregated results :\n"
    else:
        cm = results["Confusion matrix"]
        accuracy = results["Accuracy"]
        aa = results['AA']
        classacc = results["class acc"]
        kappa = results["Kappa"]

    text += "Confusion matrix :\n"
    text += str(cm)
    text += "---\n"

    if agregated:
        text += ("Accuracy: {:.02f}±{:.02f}\n".format(np.mean(accuracies),
                                                         np.std(accuracies)))
    else:
        text += "Accuracy : {:.02f}%\n".format(accuracy)
    text += "---\n"

    text += "class acc :\n"
    if agregated:
        for label, score, std in zip(label_values, class_acc_mean,
                                     class_acc_std):
            text += "\t{}: {:.02f}±{:.02f}\n".format(label, score, std)
    else:
        for label, score in zip(label_values, classacc):
            text += "\t{}: {:.02f}\n".format(label, score)
    text += "---\n"

    if agregated:
        text += ("AA: {:.02f}±{:.02f}\n".format(np.mean(aa),
                                                   np.std(aa)))
        text += ("Kappa: {:.02f}±{:.02f}\n".format(np.mean(kappas),
                                                      np.std(kappas)))
    else:
        text += "AA: {:.02f}%\n".format(aa)
        text += "Kappa: {:.02f}\n".format(kappa)
    
    text += "---\n"

    print(text)

def show_results1(results, vis, label_values=None, agregated=False):

    text = ""
    if agregated:
        accuracies = [r["Accuracy"] for r in results]
        kappas = [r["Kappa"] for r in results]
        F1_scores = [r["F1 scores"] for r in results]

        F1_scores_mean = np.mean(F1_scores, axis=0)
        F1_scores_std = np.std(F1_scores, axis=0)
        cm = np.mean([r["Confusion matrix"] for r in results], axis=0)
        text += "Agregated results :\n"
    else:
        cm = results["Confusion matrix"]
        accuracy = results["Accuracy"]
        F1scores = results["F1 scores"]
        kappa = results["Kappa"]
        class_accuracy = results["Class_Accuracy"]


    label_values = label_values[1:]

    vis.heatmap(cm, opts={'title': "Confusion matrix",
                          'marginbottom': 150,
                          'marginleft': 150,
                          'width': 500,
                          'height': 500,
                          'rownames': label_values, 'columnnames': label_values})

    text += "Confusion matrix :\n"
    text += str(cm)
    text += "---\n"

    if agregated:
        text += ("Accuracy: {:.03f} +- {:.03f}\n".format(np.mean(accuracies),
                                                         np.std(accuracies)))
    else:
        text += "Accuracy : {:.03f}%\n".format(accuracy)

        text += "class_accuracy:"
        text += str(class_accuracy)

    text += "---\n"

    text += "F1 scores :\n"
    if agregated:
        for label, score, std in zip(label_values, F1_scores_mean,
                                     F1_scores_std):
            text += "\t{}: {:.03f} +- {:.03f}\n".format(label, score, std)
    else:
        for label, score in zip(label_values, F1scores):
            text += "\t{}: {:.03f}\n".format(label, score)
    text += "---\n"

    if agregated:
        text += ("Kappa: {:.03f} +- {:.03f}\n".format(np.mean(kappas),
                                                      np.std(kappas)))
    else:
        text += "Kappa: {:.03f}\n".format(kappa)

    vis.text(text.replace('\n', '<br/>'))
    print('text is :',text)
    print('混淆矩阵中,一行求和总数是该类别的真实数量，一列求和就是该类别的预测数量')


def plot_with_labels(lowDWeights, labels, path):
    plt.cla()
    # 降到二维了，分别给x和y
    labels_num = np.max(labels)
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    area = np.pi * 2 ** 2  # 点面积
    # 遍历每个点以及对应标签
    color = []
    for i in range(labels_num):
        coo_x = np.where(labels == (i + 1))
        color = cm.rainbow(int(255/labels_num * (i + 1))) # 为了使得颜色有区分度，把0-255颜色区间分为9分,然后把标签映射到一个区间
        # print(color,)
        color = np.array(color).reshape(1,-1)
        plt.scatter(X[coo_x], Y[coo_x], s = area ,c=color,)
        print(i)

    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer')

    plt.savefig(path, dpi = 400)

def convert_to_color_(arr_2d, palette=None):
    """Convert an array of labels to RGB color-encoded image.

    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)

    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format

    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        raise Exception("Unknown color palette")
    print(palette)
    # quit()
    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_from_color_(arr_3d, palette=None):
    """Convert an RGB-encoded image to grayscale labels.

    Args:
        arr_3d: int 2D image of color-coded labels on 3 channels
        palette: dict of colors used (RGB tuple -> label number)

    Returns:
        arr_2d: int 2D array of labels

    """
    if palette is None:
        raise Exception("Unknown color palette")

    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


def convert_to_color(x, N_Classes, palette):
    
    if palette is None:
    # Generate color palette
        palette = {0: (0, 0, 0)}

    for k, color in enumerate(sns.color_palette("hls", N_Classes - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))

    invert_palette = {v: k for k, v in palette.items()}
    
    return convert_to_color_(x, palette=palette)

def display_predictions(pred, vis, gt=None, caption=""):
    # # 通过模块方式启动 python -m visdom.server -p 8888
    if gt is None:

        vis.images([np.transpose(pred, (2, 0, 1))],
                    opts={'caption': caption})
    else:

        vis.images([np.transpose(pred, (2, 0, 1)),
                    np.transpose(gt, (2, 0, 1))],
                    nrow=2,
                    opts={'caption': caption})


def explore_spectrums(img, complete_gt, class_names, vis,ignored_labels=None):
    """Plot sampled spectrums with mean + std for each class.

    Args:
        img: 3D hyperspectral image
        complete_gt: 2D array of labels
        class_names: list of class names
        ignored_labels (optional): list of labels to ignore
        vis : Visdom display
    Returns:
        mean_spectrums: dict of mean spectrum by class

    """
    mean_spectrums = {}
    for c in np.unique(complete_gt):
        if c in ignored_labels:
            continue
        mask = complete_gt == c
        class_spectrums = img[mask].reshape(-1, img.shape[-1])
        step = max(1, class_spectrums.shape[0] // 100)
        fig = plt.figure()
        plt.title(class_names[c])
        # Sample and plot spectrums from the selected class
        for spectrum in class_spectrums[::step, :]:
            plt.plot(spectrum, alpha=0.25)
        mean_spectrum = np.mean(class_spectrums, axis=0)
        std_spectrum = np.std(class_spectrums, axis=0)
        lower_spectrum = np.maximum(0, mean_spectrum - std_spectrum)
        higher_spectrum = mean_spectrum + std_spectrum

        # Plot the mean spectrum with thickness based on std
        plt.fill_between(range(len(mean_spectrum)), lower_spectrum,
                         higher_spectrum, color="#3F5D7D")
        plt.plot(mean_spectrum, alpha=1, color="#FFFFFF", lw=2)
        vis.matplot(plt)
        mean_spectrums[class_names[c]] = mean_spectrum
    return mean_spectrums

def plot_spectrums(spectrums, vis, title=""):
    """Plot the specified dictionary of spectrums.

    Args:
        spectrums: dictionary (name -> spectrum) of spectrums to plot
        vis: Visdom display
    """
    win = None
    for k, v in spectrums.items():
        n_bands = len(v)
        update = None if win is None else 'append'
        win = vis.line(X=np.arange(n_bands), Y=v, name=k, win=win, update=update,
                       opts={'title': title})
# 以上是数据可视化用到的函数

def generate_t_sne(predict_mask, masks, x, y):

    for index, (mask, x1, y1) in enumerate(zip(masks, x, y)):

        predict_mask[x1, y1, :] = mask

    return predict_mask
    

def t_sne(HSI_or_high_feature, label,save_path, show_figure=False):

    """
    :param HSI_or_high_feature: input data (HSI or high feature)---> (H,W,B)
    :param label:  input label ---> (H,W)
    :param save_path:  save figure path
    :return: figure
    """

    print(HSI_or_high_feature.shape)

    HSI_or_high_feature = HSI_or_high_feature.permute(2, 0, 1)
    print(HSI_or_high_feature.shape)

    d = np.zeros_like(label)
    d[label > 0] = True

    mask = copy.deepcopy(d)
    mask = mask.astype(bool)


    out = []
    for i in range(HSI_or_high_feature.shape[0]):

        img = HSI_or_high_feature[i, :, :]
        img = img.float()

        b = img[mask]
        b = b.reshape(1, -1)

        if i == 0:
            out = b
            continue
        out = np.concatenate((out, b), 0)

    HSI_or_high_feature = out.transpose(1, 0)
    print(HSI_or_high_feature.shape)
    label = label[mask]

    labels_num = np.max(label)
    area = np.pi * 1.5 ** 2  # 点面积

    HSI_or_high_feature_reduced = TSNE(n_components=2, init="pca", random_state=0).fit_transform(HSI_or_high_feature)

    for i in range(labels_num):
        print(i)
        coo_x = np.where(label == (i + 1))

        color_one = cm.rainbow(int(255 / labels_num * (i + 1)))  # 为了使得颜色有区分度，把0-255颜色区间分为9分,然后把标签映射到一个区间
        color_ = np.array(color_one).reshape(1, -1)

        plt.scatter(HSI_or_high_feature_reduced[coo_x, 0], HSI_or_high_feature_reduced[coo_x, 1], marker='o', s=area, c=color_)

    plt.box(False)

# 去掉X轴和Y轴的刻度
    plt.xticks([])
    plt.yticks([])
    plt.savefig(save_path, dpi=400)

    if not show_figure:

        plt.show()

    print("Visualization is complete")
