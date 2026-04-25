import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, multilabel_confusion_matrix, accuracy_score, balanced_accuracy_score, hamming_loss,average_precision_score
import math

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')       #  np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
########################################################################
## 函数的作用是对异常检测的预测结果 (pred) 进行调整，以更好地匹配真实标签 (gt)
## 核心逻辑：当检测到一个异常点时，会向前和向后扩展预测结果，以确保连续的异常段被完整标记
##
##
########################################################################
def adjustment(gt, pred):
    anomaly_state = False  # 初始化一个标志位，用于指示当前是否处于异常段中
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
        # 如果当前数据点的真实标签 (gt[i]) 和预测结果 (pred[i]) 都为 1（表示异常），并且当前不处于异常段中（not anomaly_state），则进入异常段处理逻辑。
            anomaly_state = True  # 当前处于异常段中
            for j in range(i, 0, -1):  # 向前扩展预测结果
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):  # 向后扩展预测结果
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        # 如果当前处于异常段中，则将当前数据点的预测结果强制设置为 1（异常）
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def cal_metrics(y_pred, y_true):
    """
        计算以下指标：
        - AUC
        - ACC
        - Specificity
        - Sensitivity (Recall)
        - PPV (Positive Predictive Value, Precision)
        - NPV (Negative Predictive Value)
    """
    # 确保输入是 numpy 数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 如果输入是logits，转换为概率
    if y_pred.min() < 0 or y_pred.max() > 1:  # 判断是否是logits
        y_prob = 1 / (1 + np.exp(-y_pred))  # sigmoid转换
    else:
        y_prob = y_pred

    y_pred_binary = (y_prob > 0.5).astype(int)  # 二值化预测

    n_classes = y_true.shape[1]
    results = {
        'per_class': {},
        'overall': {}
    }

    # 计算每个类别的指标
    for i in range(n_classes):
        try:
            roc_auc = roc_auc_score(y_true[:, i], y_prob[:, i])
            # pr_auc = average_precision_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            auc = np.nan  # 当类别只有一个样本时设为NaN

        results['per_class'][f'class_{i}'] = {
            'accuracy': accuracy_score(y_true[:, i], y_pred_binary[:, i]),
            'precision': precision_score(y_true[:, i], y_pred_binary[:, i], zero_division=0),
            'recall': recall_score(y_true[:, i], y_pred_binary[:, i], zero_division=0),
            'f1': f1_score(y_true[:, i], y_pred_binary[:, i], zero_division=0),
            'balance_acc': balanced_accuracy_score(y_true[:, i], y_pred_binary[:, i], adjusted=False),
            'roc_auc': roc_auc,
        }

    # balanced_accuracy_score(y_true, y_pred, adjusted=False)
    # hamming_loss(y_true, y_pred)
    # 计算整体指标（微平均）
    results['overall'] = {
        'micro_accuracy': accuracy_score(y_true.ravel(), y_pred_binary.ravel()),
        'micro_precision': precision_score(y_true, y_pred_binary, average='micro', zero_division=0),
        'micro_recall': recall_score(y_true, y_pred_binary, average='micro', zero_division=0),
        'micro_f1': f1_score(y_true, y_pred_binary, average='micro', zero_division=0),
        'macro_auc_roc': np.nanmean([v['roc_auc'] for v in results['per_class'].values()]),
        'balance_acc_all': np.nanmean([v['balance_acc'] for v in results['per_class'].values()]),
        'hamming_loss': hamming_loss(y_true, y_pred)
    }
    # multilabel_confusion_matrix(y_true, y_pred_binary)

    return results

    # # 计算 AUC: ROC曲线下的面积，衡量分类器的整体性能
    # auc = roc_auc_score(y_true, y_pred)
    #
    # # 将预测值转换为二分类（0 或 1）
    # y_pred_binary = np.round(y_pred).astype(int)
    #
    # # 计算混淆矩阵
    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    #
    # # 计算 Specificity: 模型正确识别负类的能力
    # specificity = tn / (tn + fp)
    #
    # # 计算 Sensitivity (Recall): 表示模型正确识别正类的能力
    # sensitivity = tp / (tp + fn)
    #
    # # 计算 PPV (Precision): 表示模型预测为正类的样本中实际为正类的比例
    # ppv = tp / (tp + fp)
    #
    # # 计算 NPV: 表示模型预测为负类的样本中实际为负类的比例
    # npv = tn / (tn + fn)
    #
    # # 返回所有指标
    # return {
    #     "AUC": auc,
    #     "Specificity": specificity,
    #     "Sensitivity": sensitivity,
    #     "PPV": ppv,
    #     "NPV": npv
    # }
