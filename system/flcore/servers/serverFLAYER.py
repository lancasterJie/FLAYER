import copy
import numpy as np
import torch
import time
import openpyxl as op
import random
from typing import Dict, List, Optional, Tuple
from functools import reduce
from collections import OrderedDict
from flcore.clients.clientFLAYER import *
from utils.data_utils import read_client_data
from threading import Thread

class FLAYER(object):
    def __init__(self, args, times):
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.join_clients = int(self.num_clients * self.join_ratio)

        self.clients = []
        self.selected_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        self.aggregate_params = []

        self.rs_test_acc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap

        self.set_clients(args, clientFLAYER)

        self.wb = op.Workbook()
        self.ws = self.wb['Sheet']

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

    def train(self):
        # best_acc = 0.0
        accs = [0.0] * self.num_clients
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models(accs)

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                accs, test_acc = self.evaluate(nonprint=None)


            for client in self.selected_clients:
                client.train()

            # 保存模型
            # if i == 1 or i == 25:
            #     self.save_models(i)

            # if test_acc > best_acc:
            #     self.save_models()
            #     best_acc = test_acc

            # if i % self.eval_gap == 0:
            #     # print(f"\n-------------Round number: {i}-------------")
            #     # print("\nEvaluate global model")
            #     accs = self.evaluate(i, nonprint=None)

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models(accs)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])

        print("\nBest global accuracy.")
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

    def set_clients(self, args, clientObj):
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(args,
                               id=i,
                               train_samples=len(train_data),
                               test_samples=len(test_data))
            self.clients.append(client)

    def select_clients(self):
        if self.random_join_ratio:
            join_clients = np.random.choice(range(self.join_clients, self.num_clients + 1), 1, replace=False)[0]
        else:
            join_clients = self.join_clients
        selected_clients = list(np.random.choice(self.clients, join_clients, replace=False))

        return selected_clients


    def send_models(self, accs):
        assert (len(self.clients) > 0)

        for client, acc in zip(self.clients, accs):
            client.local_initialization(self.global_model, acc)

    def save_models(self, i=None):
        for client in self.clients:
            client.save_models(i)

    def receive_models(self, accs):
        assert (len(self.selected_clients) > 0)

        self.aggregate_params = []
        self.uploaded_ids = []

        s_t = time.time()
        for client, acc in zip(self.selected_clients, accs):
            self.uploaded_ids.append(client.id)
            self.aggregate_params.append((client.get_parameters_sparse(client.model_before, client.model),
                                          client.train_samples))

        print("mask:")
        print('-' * 50, time.time() - s_t)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def aggregate_parameters(self):

        parameters_lastround = self.get_parameters(self.global_model)

        s_t = time.time()
        parameters_thisround_sparse_np = aggregate_sparse(self.aggregate_params)

        print("aggregate:")
        print('-' * 50, time.time() - s_t)

        s_t1 = time.time()
        parameters_thisround = [np.where(layer == 0, Layer, layer)
                                for layer, Layer in zip(parameters_thisround_sparse_np, parameters_lastround)]
        print("fill zero:")
        print('-' * 50, time.time() - s_t1)

        self.set_parameters(self.global_model, parameters_thisround)

        del parameters_lastround, parameters_thisround, parameters_thisround_sparse_np

    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        accs = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            # print(f'Client {c.id}: Acc: {ct*1.0/ns}, AUC: {auc}')
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)
            accs.append(ct * 1.0 / ns)
        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc, accs

    def train_metrics(self):
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            # print(f'Client {c.id}: Train loss: {cl*1.0/ns}')
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    def evaluate(self, acc=None, loss=None, nonprint=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        losses = [a / n for a, n in zip(stats_train[2], stats_train[1])]

        # 每轮保存这个准确率.
        # data = []
        # data.append(test_acc)
        # data += accs
        # self.ws.append(data)
        # filename = "cifar100_cnn_flayer.xlsx"
        # self.wb.save(filename)

        if nonprint == None:
            if acc == None:
                self.rs_test_acc.append(test_acc)
            else:
                acc.append(test_acc)

            if loss == None:
                self.rs_train_loss.append(train_loss)
            else:
                loss.append(train_loss)

            print("Averaged Train Loss: {:.4f}".format(train_loss))
            print("Averaged Test Accurancy: {:.4f}".format(test_acc))
            print("Averaged Test AUC: {:.4f}".format(test_auc))
            print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
            print("Std Test AUC: {:.4f}".format(np.std(aucs)))
        # else:
        #     return accs
        return accs, test_acc
        # return stats[4]

    def set_parameters(self, model, parameters):
        # 注意dtype，曾在tiny数据集上报dtype错误
        for new_param, old_param in zip(parameters, model.parameters()):
            old_param.data = torch.tensor(new_param, dtype=torch.float).to(self.device)
            # old_param.data = torch.tensor(new_param).to(self.device)

    def get_parameters(self, model):
        return [val.data.cpu().numpy() for val in model.parameters()]

def aggregate_sparse(results):
    num_examples_total = sum([num_examples for _, num_examples in results])

    weighted_weights = [
        np.multiply(weights, num_examples) for weights, num_examples in results
    ]

    client_num_examples = np.array([num_examples for _, num_examples in results])
    weights_prime = []

    for layer_updates in zip(*weighted_weights):

        # if layer_updates[0].ndim != 0:  # Assuming this is for conv2d layers ==4
        if layer_updates[0].ndim == 4 or layer_updates[0].ndim == 1:
        # if layer_updates[0].ndim == 4:
            weight_matrix = np.add.reduce(layer_updates)
            num_total_matrix = np.full_like(layer_updates[0], num_examples_total)

            for client_id, layer in enumerate(layer_updates):
                num_total_matrix[layer == 0] -= client_num_examples[client_id]

            weights_prime.append(np.divide(weight_matrix, num_total_matrix,
                                           out=np.zeros_like(weight_matrix), where=num_total_matrix != 0))
        else:
            # For fully connected layers or similar
            weight_matrix = np.add.reduce(layer_updates)
            weights_prime.append(weight_matrix / num_examples_total)

            # weights_prime.append(reduce(np.add, layer_updates) / num_examples_total)

            # weights_prime.append(np.sum(layer_updates, axis=0) / num_examples_total)

    return weights_prime

