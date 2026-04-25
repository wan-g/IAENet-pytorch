from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy, cal_metrics
from utils.loss import FocalBCELoss, AsymmetricLoss, PolyLoss, FocalBCELossv1, PriorWeightedBCELoss, LCRLoss
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        vali_data, vali_loader = self._get_data(flag='VAL')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len, vali_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        # self.args.num_class = len(train_data.class_names)
        self.args.num_class = train_data.class_names
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.BCEWithLogitsLoss()
        return criterion

    def get_adaptive_weights(self, targets, eps=1e-8):
        """
        targets: Tensor of shape [batch_size, num_classes], with 0/1 labels
        Return: pos_weight
        """
        pos_counts = targets.sum(dim=0) + eps  # shape: [num_classes]
        neg_counts = (1 - targets).sum(dim=0) + eps
        total = pos_counts + neg_counts

        label_priors = pos_counts / total

        return label_priors

    def compute_global_co_matrix(self, train_loader, num_classes, device='cpu', max_batches=256):
        """

        """
        co_matrix = torch.zeros(num_classes, num_classes).to(device)

        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                # if i >= max_batches:
                #     break
                labels = batch[1].to(device).float()  # shape: [B, C]    # (batch_x, label, padding_mask)
                co_matrix += torch.matmul(labels.squeeze(1).T, labels.squeeze(1))  # [C, C]

        # 归一化（例如最大归一化或行归一化）
        co_matrix = co_matrix / (co_matrix.max() + 1e-8)

        return co_matrix


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)

                label = label.squeeze(1).float().to(self.device)   # 多标签分类任务中，label 是浮点型

                outputs = self.model(batch_x, padding_mask, None, None)


                if self.args.loss_strategy == 'FocalBCELoss':
                    loss_fn = FocalBCELoss()
                elif self.args.loss_strategy == 'AsymmetricLoss':
                    loss_fn = AsymmetricLoss()
                elif self.args.loss_strategy == 'BCELoss' or 'LCRLoss':
                    loss_fn = nn.BCEWithLogitsLoss()
                elif self.args.loss_strategy == 'MarginLoss':
                    loss_fn = nn.MultiLabelMarginLoss()
                elif self.args.loss_strategy == 'SoftMarginLoss':
                    loss_fn = nn.MultiLabelSoftMarginLoss()
                elif self.args.loss_strategy == 'PolyLoss':
                    loss_fn = PolyLoss()
                elif self.args.loss_strategy == 'PriorWeightedBCELoss':
                    loss_fn = PriorWeightedBCELoss(
                        label_priors=torch.tensor([47048 / 4878325, 119930 / 4878325, 137909 / 4878325, 9741 / 4878325,
                        83434 / 4878325, 20051 / 4878325], dtype=torch.float32).to(self.device), weight_strategy=self.args.weight_strategy
                    ).to(self.device)

                loss = loss_fn(outputs, label.squeeze(1))  # long.
                total_loss.append(loss.item())

                preds.append(outputs.detach())
                # print(preds)
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        # print(preds)
        trues = torch.cat(trues, 0)
        # probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        # predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        probs = torch.sigmoid(preds)  # 将 logits 转换为概率
        predictions = (probs > 0.5).float()
        metrics = cal_metrics(predictions.cpu().numpy(), trues.cpu().numpy())

        self.model.train()
        return total_loss, metrics

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='VAL')  # TODO
        test_data, test_loader = self._get_data(flag='TEST')


        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # Initialize lists to store loss values
        train_loss_history = []
        val_loss_history = []
        test_loss_history = []
        val_acc_history = []
        test_acc_history = []

        co_matrix = self.compute_global_co_matrix(train_loader, num_classes=self.args.num_class).to(self.device)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()


            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                # label = label.to(self.device)
                label = label.float().to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                if self.args.prior == 'local':
                    priors = self.get_adaptive_weights(targets=label.squeeze(1))
                elif self.args.prior == 'golbal':
                    priors = torch.tensor([47048 / 4878325, 119930 / 4878325, 137909 / 4878325, 9741 / 4878325,
                        83434 / 4878325, 20051 / 4878325], dtype=torch.float32).to(self.device) #根据数据集情况进行修改

                if self.args.loss_strategy == 'FocalBCELoss':
                    loss_fn = FocalBCELoss()
                elif self.args.loss_strategy == 'AsymmetricLoss':
                    loss_fn = AsymmetricLoss()
                elif self.args.loss_strategy == 'BCELoss':
                    loss_fn = nn.BCEWithLogitsLoss()
                elif self.args.loss_strategy == 'MarginLoss':
                    loss_fn = nn.MultiLabelMarginLoss()   # 不能直接用常见的 multi-hot 形式
                elif self.args.loss_strategy == 'SoftMarginLoss':
                    loss_fn = nn.MultiLabelSoftMarginLoss()
                elif self.args.loss_strategy == 'PolyLoss':
                    loss_fn = PolyLoss()
                elif self.args.loss_strategy == 'PriorWeightedBCELoss':
                    loss_fn = PriorWeightedBCELoss(
                        label_priors=priors, weight_strategy=self.args.weight_strategy
                    ).to(self.device)
                elif self.args.loss_strategy == 'LCRLoss':
                    loss_fn = LCRLoss(
                        label_priors=priors, weight_strategy=self.args.weight_strategy,
                        lambda_co=self.args.lambda_co, co_matrix=co_matrix
                    ).to(self.device)


                if self.args.loss_strategy == 'LCRLoss':
                    loss = loss_fn(outputs, label.squeeze(1))
                else:
                    loss = loss_fn(outputs, label.squeeze(1))  #long().

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_metrics = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_metrics = self.vali(test_data, test_loader, criterion)

            # Store the loss and accuracy values
            train_loss_history.append(train_loss)
            val_loss_history.append(vali_loss)
            test_loss_history.append(test_loss)
            val_acc_history.append(val_metrics['overall']['micro_accuracy'])
            test_acc_history.append(test_metrics['overall']['micro_accuracy'])

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_metrics['overall']['micro_accuracy'], test_loss, test_metrics['overall']['micro_accuracy']))
            early_stopping(-val_metrics['overall']['micro_f1'], self.model, path)  # TODO
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # Plot the training and validation loss curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss_history, label='Training Loss')
        plt.plot(val_loss_history, label='Validation Loss')
        plt.plot(test_loss_history, label='Test Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(path, 'loss_curve.png'))
        plt.close()

        # Plot the accuracy curves
        plt.figure(figsize=(10, 5))
        plt.plot(val_acc_history, label='Validation Accuracy')
        plt.plot(test_acc_history, label='Test Accuracy')
        plt.title('Accuracy Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(path, 'accuracy_curve.png'))
        plt.close()

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                # label = label.to(self.device)
                label = label.squeeze(1).float().to(self.device)  # 多标签分类任务中，label 是浮点型

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label.squeeze(1))

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        # probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        # predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        probs = torch.sigmoid(preds)
        predictions = (probs > 0.5).float()  # TODO
        # trues = trues.flatten().cpu().numpy()
        # accuracy = cal_accuracy(predictions, trues)
        # accuracy = cal_accuracy(predictions.cpu().numpy(), trues.cpu().numpy())
        metrics = cal_metrics(predictions.cpu().numpy(), trues.cpu().numpy())

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 打印结果
        print("=== 每个类别指标 ===")
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"{class_name}:")
            print(f"  准确率: {class_metrics['accuracy']:.4f}")
            print(f"  精确率: {class_metrics['precision']:.4f}")
            print(f"  召回率: {class_metrics['recall']:.4f}")
            print(f"  F1分数: {class_metrics['f1']:.4f}")
            print(f"  ROC_AUC: {class_metrics['roc_auc']:.4f}")
            print(f"平均ACC: {class_metrics['balance_acc']:.4f}")

        print("\n=== 整体指标 ===")
        print(f"微平均准确率: {metrics['overall']['micro_accuracy']:.4f}")
        print(f"微平均精确率: {metrics['overall']['micro_precision']:.4f}")
        print(f"微平均召回率: {metrics['overall']['micro_recall']:.4f}")
        print(f"微平均F1: {metrics['overall']['micro_f1']:.4f}")
        print(f"宏平均ROC_AUC: {metrics['overall']['macro_auc_roc']:.4f}")
        print(f"Balance_acc_all: {metrics['overall']['balance_acc_all']:.4f}")
        print(f"Hamming loss: {metrics['overall']['hamming_loss']:.4f}")


        # print('accuracy:{}'.format(accuracy))
        file_name = 'result_classification.txt'
        f = open(os.path.join(folder_path, file_name), 'a')
        f.write(setting + "  \n")
        f.write('metrics:{}'.format(metrics))
        f.write('\n')
        f.write('\n')
        f.close()
        return
