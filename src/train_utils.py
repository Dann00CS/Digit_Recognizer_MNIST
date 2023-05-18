import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

def plot_losses(loss, acc):
    fig, ax = plt.subplots(1,2, figsize=(15,5))
    ax[0].plot(loss["train"], label="train")
    ax[0].plot(loss["val"], label="val")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[1].plot(acc["val"], label="val")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()
    plt.show()

def get_and_plot_matrix_confusion(y_preds, labels):
    conf_matrix = confusion_matrix(y_preds, labels)
    fig, ax = plt.subplots(1,1,figsize=(15,15))
    sns.heatmap(conf_matrix, annot=True, fmt="g", ax=ax)
    ax.set(xlabel='Predicted labels', ylabel='True labels')
    plt.show()
    return conf_matrix

def get_worst_classes(conf_matrix):
    def get_num_wrong_preds_each_class(conf_matrix_arr):
        num_wrong_preds = {}
        for i in range(conf_matrix_arr.shape[0]):
            row = conf_matrix_arr[i].copy()
            row[i] = 0
            num_wrong_preds[i] = np.sum(row)
        return num_wrong_preds
    wrong_preds_dict = get_num_wrong_preds_each_class(conf_matrix)
    ordered_wrong_preds_dict = {k: v for k, v in sorted(wrong_preds_dict.items(), key=lambda item: item[1], reverse=True)}
    print(ordered_wrong_preds_dict)
    plt.plot(wrong_preds_dict.values(), marker="o")
    plt.xticks([i for i in range(0,10)])
    plt.show()
    worst_classes = [k for k,v in wrong_preds_dict.items() if v>= np.mean(list(wrong_preds_dict.values()))]
    return worst_classes