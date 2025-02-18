import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
from utils.FLAYER_aggregation import LocalAggregation
from multiprocessing import cpu_count
import openpyxl as op
import math
import os

class clientFLAYER(object):
    def __init__(self, args, id, train_samples, test_samples):
        self.model = copy.deepcopy(args.model)
        self.model_before = None
        self.dataset = args.dataset
        self.device = args.device
        self.id = id
        self.args = args

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps

        self.loss = nn.CrossEntropyLoss()
        self.layer_idx = args.layer_idx

        self.wb = op.Workbook()
        self.ws = self.wb['Sheet']

        if args.model_str == "cnn":
            # CNN
            params = [
                {'params': list(self.model.parameters())[:2], 'lr': self.learning_rate},
                {'params': list(self.model.parameters())[2:4], 'lr': self.learning_rate * 30},
                {'params': list(self.model.parameters())[4:6], 'lr': self.learning_rate * 40},
                {'params': list(self.model.parameters())[6:8], 'lr': self.learning_rate / 5},
            ]

        elif args.model_str == "resnet":
            # Resnet
            params = [
                {'params': self.model.conv1.parameters(), 'lr': self.learning_rate * 1},
                {'params': self.model.bn1.parameters(), 'lr': self.learning_rate*1.5},
                {'params': self.model.layer1.parameters(), 'lr': self.learning_rate*2},
                {'params': self.model.layer2.parameters(), 'lr': self.learning_rate*2.5},
                {'params': self.model.layer3.parameters(), 'lr': self.learning_rate*3},
                {'params': self.model.layer4.parameters(), 'lr': self.learning_rate*3.5},
                {'params': self.model.fc.parameters(), 'lr': self.learning_rate / 10}
            ]

        elif args.model_str == "fastText":
            # fastText
            params = [
                {'params': list(self.model.parameters())[:1], 'lr': self.learning_rate},
                {'params': list(self.model.parameters())[1:3], 'lr': self.learning_rate * 2},
                {'params': list(self.model.parameters())[3:5], 'lr': self.learning_rate / 10},
            ]

        self.optimizer = torch.optim.SGD(params)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.local_aggregation = LocalAggregation(self.layer_idx)


    def train(self):
        trainloader = self.load_train_data()
        self.model_before = copy.deepcopy(self.model)
        self.model.train()

        for step in range(self.local_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()

                # lr formula each big layer
                if i == 0:
                    adaptive_lr = []
                    idx = 0
                    for name, layer in self.model.named_children():
                        grads = []
                        for param in layer.parameters():
                            if param.grad is not None:
                                grad_norm = param.grad.data.norm(2).item()
                                if self.args.model_str == "cnn":
                                    grads.append(20 * self.learning_rate *
                                                 (1 + (idx / 3) * math.log(1 + 1 / grad_norm)))  #  + 1e-8

                                elif self.args.model_str == "resnet":
                                    grads.append(self.learning_rate *
                                                 (1 + (idx / 6) * math.log(1 + 1 / grad_norm)))  # + 1e-8

                                elif self.args.model_str == "fastText":
                                    grads.append(self.learning_rate *
                                                 (1 + (idx / 2) * math.log(1 + 1 / grad_norm)))  # + 1e-8

                        if grads:
                            idx += 1
                            adaptive_lr.append(sum(grads) / len(grads))



                    if self.args.model_str == "cnn":
                        params = [
                            {'params': list(self.model.parameters())[:2], 'lr': self.learning_rate},
                            {'params': list(self.model.parameters())[2:4], 'lr': adaptive_lr[1]},
                            {'params': list(self.model.parameters())[4:6], 'lr': adaptive_lr[2]},
                            {'params': list(self.model.parameters())[6:8], 'lr': self.learning_rate / 5},
                        ]

                    elif self.args.model_str == "resnet":
                        params = [
                            {'params': self.model.conv1.parameters(), 'lr': self.learning_rate},
                            {'params': self.model.bn1.parameters(), 'lr': adaptive_lr[1]},
                            {'params': self.model.layer1.parameters(), 'lr': adaptive_lr[2]},
                            {'params': self.model.layer2.parameters(), 'lr': adaptive_lr[3]},
                            {'params': self.model.layer3.parameters(), 'lr': adaptive_lr[4]},
                            {'params': self.model.layer4.parameters(), 'lr': adaptive_lr[5]},
                            {'params': self.model.fc.parameters(), 'lr': self.learning_rate / 5}
                        ]

                    elif self.args.model_str == "fastText":
                        params = [
                            {'params': list(self.model.parameters())[:1], 'lr': self.learning_rate},
                            {'params': list(self.model.parameters())[1:3], 'lr': adaptive_lr[1]},
                            {'params': list(self.model.parameters())[3:5], 'lr': self.learning_rate / 5},
                        ]

                    self.optimizer = torch.optim.SGD(params)

                self.optimizer.step()


    def local_initialization(self, received_global_model, acc):
        self.local_aggregation.adaptive_local_aggregation(received_global_model, self.model, acc)

        trainloader = self.load_train_data()
        self.model.train()
        params = list(self.model.parameters())
        for param in params[:-self.layer_idx]:
            param.requires_grad = False
        # for param in params[-self.layer_idx:]:
        #     param.requires_grad = True

        for step in range(1):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

        for param in params[:-self.layer_idx]:
            param.requires_grad = True

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=False)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=True, shuffle=False)

    def test_metrics(self, model=None):
        testloader = self.load_test_data()
        if model == None:
            model = self.model
        model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(F.softmax(output).detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc

    def train_metrics(self, model=None):
        trainloader = self.load_train_data()
        if model == None:
            model = self.model
        model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    def get_parameters_sparse(self, model_before, model):

        model_param = [val.data.cpu().numpy() for val in model.parameters()]
        model_change = [np.abs(val1.data.detach().cpu().numpy() - val2.data.detach().cpu().numpy())
                        for val1, val2 in zip(model.parameters(), model_before.parameters())]

        layer_len = len(model_param)
        for layer_id, (change_layer, param_layer) in enumerate(zip(model_change, model_param)):

            mask_ratio = 1 - max(min(layer_id / layer_len, 0.9), 0.1)
            if ((layer_len - 2) <= layer_id < layer_len):
                mask_ratio = 0.0

            # Check if the layer should be processed
            if mask_ratio != 0.0:
                if change_layer.ndim == 4: #and change_layer.shape[2] != 1:
                    # Compute prune threshold
                    kernel_num = change_layer.shape[2] ** 2
                    # prune = np.round(kernel_num * mask_ratio).astype(int)
                    if mask_ratio >= (1 / kernel_num) and mask_ratio <= ((kernel_num - 1) / kernel_num):
                        prune = np.round(kernel_num * mask_ratio).astype(int)
                    elif mask_ratio < (1 / kernel_num):
                        prune = 1
                    else:
                        prune = kernel_num - 1

                    # Reshape the layer for easier processing
                    reshaped_layer = change_layer.reshape(change_layer.shape[0], change_layer.shape[1], -1)
                    # Sort each filter of each output channel, keep only the indices
                    sorted_indices = np.argsort(reshaped_layer, axis=-1)

                    # Determine the threshold index for each filter of each output channel
                    threshold_indices = sorted_indices[:, :, :prune]

                    # Create a mask for elements to zero out
                    mask = np.ones_like(reshaped_layer, dtype=bool)
                    np.put_along_axis(mask, threshold_indices, False, axis=-1)

                    # Apply mask to the original parameter layer, after reshaping the mask back
                    # param_layer[mask.reshape(param_layer.shape)] = 0
                    param_layer *= (mask.reshape(param_layer.shape))

                elif change_layer.ndim == 1:
                    element_num = change_layer.shape[0]

                    if mask_ratio >= (1 / element_num) and mask_ratio <= ((element_num - 1) / element_num):
                        prune = np.round(element_num * mask_ratio).astype(int)
                    elif mask_ratio < (1 / element_num):
                        prune = 1
                    else:
                        prune = element_num - 1

                    sorted_indices = np.argsort(change_layer, axis=-1)
                    threshold_indices = sorted_indices[:prune]
                    param_layer[threshold_indices] = 0

                # if change_layer.ndim == 2 and self.args.model_str == "fastText" and layer_id >= 1:
                #     element_num = change_layer.shape[1]
                #
                #     if mask_ratio >= (1 / element_num) and mask_ratio <= ((element_num - 1) / element_num):
                #         prune = np.round(element_num * mask_ratio).astype(int)
                #     elif mask_ratio < (1 / element_num):
                #         prune = 1
                #     else:
                #         prune = element_num - 1
                #
                #     sorted_indices = np.argsort(change_layer, axis=-1)
                #
                #     # Determine the threshold index for each filter of each output channel
                #     threshold_indices = sorted_indices[:, :prune]
                #
                #     # Create a mask for elements to zero out
                #     mask = np.ones_like(change_layer, dtype=bool)
                #     np.put_along_axis(mask, threshold_indices, False, axis=-1)
                #
                #     # Apply mask to the original parameter layer, after reshaping the mask back
                #     param_layer *= mask

        return model_param

    # save models to support other experiments
    # def save_models(self, i=None):
    #
    #     # 初始模型
    #     if i == 1:
    #         save_path = "saves/correct/FLAYER_start_resnet/"
    #         if not os.path.exists(save_path):
    #             os.makedirs(save_path)
    #         torch.save(self.model.state_dict(), save_path + "client" + str(self.id) + "_bestmodel.pth")
    #
    #     # 中间时候的模型
    #     if i == 25:
    #         save_path = "saves/correct/FLAYER_middle_resnet/"
    #         if not os.path.exists(save_path):
    #             os.makedirs(save_path)
    #         torch.save(self.model.state_dict(), save_path + "client" + str(self.id) + "_bestmodel.pth")
    #
    #     save_path = "saves/correct/FLAYER_best_resnet/"
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     torch.save(self.model.state_dict(), save_path + "client" + str(self.id) + "_bestmodel.pth")