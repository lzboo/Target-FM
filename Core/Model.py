# -*- coding:utf-8 -*-
# Author:Bravo
# Data:2022/9/1423:26

import torch
import torch.nn as nn
import torch.nn.functional as F
import fm
import json
bar_format = '{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'


# Load RNA-FM model
fm_model, alphabet = fm.pretrained.rna_fm_t12()
batch_converter = alphabet.get_batch_converter()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fm_model = fm_model.to(device)
fm_model.eval()  # disables dropout for deterministic results

def get_embed(batch_tokens):
    # batch_tokens =batch_tokens.squeeze(0)
    with torch.no_grad():
        batch_tokens = batch_tokens.to(device)
        results = fm_model(batch_tokens, repr_layers=[12])
    token_embeddings = results["representations"][12]
    return token_embeddings


""" 网络超参设置, filters/kernel """


class HyperParam:
    def __init__(self, filters=None, kernels=None, model_json=None):
        self.dictionary = dict()
        self.name_postfix = str()

        if (filters is not None) and (kernels is not None) and (model_json is None):
            for i, (f, k) in enumerate(zip(filters,
                                           kernels)):  # get the elements of multiple lists and indexes https://note.nkmk.me/en/python-for-enumerate-zip/
                setattr(self, 'f{}'.format(i + 1), f)
                setattr(self, 'k{}'.format(i + 1), k)
                self.dictionary.update({'f{}'.format(i + 1): f, 'k{}'.format(i + 1): k})
            self.len = i + 1

            for key, value in self.dictionary.items():
                self.name_postfix = "{}_{}-{}".format(self.name_postfix, key, value)
        elif model_json is not None:
            self.dictionary = json.loads(model_json
                                         )
            for i, (key, value) in enumerate(self.dictionary.items()):
                setattr(self, key, value)
                self.name_postfix = "{}_{}-{}".format(self.name_postfix, key, value)
            self.len = (i + 1) // 2

    def __len__(self):
        return self.len


class TargetFM1(nn.Module):
    def __init__(self, hparams=None, hidden_units=30, input_shape=(2, 30), name_prefix="model"):
        super(TargetFM1, self).__init__()

        if hparams is None:
            filters, kernels = [32, 16, 64, 16], [3, 3, 3, 3]
            hparams = HyperParam(filters, kernels)
        self.name = "{}{}".format(name_prefix, hparams.name_postfix)
        self.fc_mi = nn.Linear(640, 32)
        self.fc_mr = nn.Linear(640, 32)

        if (isinstance(hparams, HyperParam)) and (len(hparams) == 4):
            self.embd1 = nn.Conv1d(4, hparams.f1, kernel_size=hparams.k1, padding=((hparams.k1 - 1) // 2))

            self.conv2 = nn.Conv1d(hparams.f1 * 2, hparams.f2, kernel_size=hparams.k2)
            self.conv3 = nn.Conv1d(hparams.f2, hparams.f3, kernel_size=hparams.k3)
            self.conv4 = nn.Conv1d(hparams.f3, hparams.f4, kernel_size=hparams.k4)

            """ out_features = ((in_length - kernel_size + (2 * padding)) / stride + 1) * out_channels """
            # flat_features = self.forward(torch.randint(1, 5, input_shape).to(device), torch.randint(1, 5, input_shape).to(device), flat_check=False)
            self.fc1 = nn.Linear(384, hidden_units)
            self.fc2 = nn.Linear(hidden_units, 2)
        else:
            raise ValueError("not enough hyperparameters")

    def forward(self, x_mirna, x_mrna, flat_check=False):
        mi_out = get_embed(x_mirna)
        mi_out = self.fc_mi(mi_out).transpose(1, 2)
        mr_out = get_embed(x_mrna)
        mr_out = self.fc_mr(mr_out).transpose(1, 2)
        # import pdb;pdb.set_trace()
        h_mirna = F.relu(mi_out)  # torch.Size([32, 32, 30])
        # print(h_mirna.shape)
        h_mrna = F.relu(mr_out)  # torch.Size([32, 32, 30])
        # print(h_mrna.shape)
        h = torch.cat((h_mirna, h_mrna), dim=1)  # torch.Size([32, 64, 30])
        # print(h.shape)
        h = F.relu(self.conv2(h))  # torch.Size([32, 16, 28])
        # print(h.shape)
        h = F.relu(self.conv3(h))  # torch.Size([32, 64, 26])
        # print(h.shape)
        h = F.relu(self.conv4(h))  # torch.Size([32, 16, 24])
        # print(h.shape)
        h = h.view(h.size(0), -1)  # torch.Size([32, 384])
        # print(h.shape)
        if flat_check:
            return h.size(1)
        h = self.fc1(h)  # torch.Size([32, 30])
        # print(h.shape)
        # y = F.softmax(self.fc2(h), dim=1)
        y = self.fc2(h)  # torch.Size([32, 2])
        # print(y.shape)

        return y

class TargetFM2(nn.Module):
    def __init__(self, hparams=None, hidden_units=30, input_shape=(2, 30), name_prefix="model"):
        super(TargetFM2, self).__init__()

        if hparams is None:
            filters, kernels = [32, 16, 64, 16], [3, 3, 3, 3]
            hparams = HyperParam(filters, kernels)
        self.name = "{}{}".format(name_prefix, hparams.name_postfix)
        self.fc_mi = nn.Linear(640, 32)
        self.fc_mr = nn.Linear(640, 32)

        if (isinstance(hparams, HyperParam)) and (len(hparams) == 4):
            self.embd1 = nn.Conv1d(4, hparams.f1, kernel_size=hparams.k1, padding=((hparams.k1 - 1) // 2))

            self.conv2 = nn.Conv1d(hparams.f1 * 2, hparams.f2, kernel_size=hparams.k2)
            self.conv3 = nn.Conv1d(hparams.f2, hparams.f3, kernel_size=hparams.k3)
            self.conv4 = nn.Conv1d(hparams.f3, hparams.f4, kernel_size=hparams.k4)

            """ out_features = ((in_length - kernel_size + (2 * padding)) / stride + 1) * out_channels """
            # flat_features = self.forward(torch.randint(1, 5, input_shape).to(device), torch.randint(1, 5, input_shape).to(device), flat_check=False)
            self.fc1 = nn.Linear(1920, hidden_units)
            self.fc2 = nn.Linear(hidden_units, 2)
        else:
            raise ValueError("not enough hyperparameters")

    def forward(self, x_mirna, x_mrna, flat_check=False):
        mi_out = get_embed(x_mirna)
        mi_out = self.fc_mi(mi_out).transpose(1, 2)
        mr_out = get_embed(x_mrna)
        mr_out = self.fc_mr(mr_out).transpose(1, 2)
        # import pdb;pdb.set_trace()
        h_mirna = F.relu(mi_out)  # torch.Size([32, 32, 30])
        # print(h_mirna.shape)
        h_mrna = F.relu(mr_out)  # torch.Size([32, 32, 30])
        # print(h_mrna.shape)
        h = torch.cat((h_mirna, h_mrna), dim=1)  # torch.Size([32, 64, 30])
        # print(h.shape)

        h = h.view(h.size(0), -1)  # torch.Size([32, 1920])
        print(h.shape)
        if flat_check:
            return h.size(1)
        h = self.fc1(h)  # torch.Size([32, 30])
        # print(h.shape)
        # y = F.softmax(self.fc2(h), dim=1)
        y = self.fc2(h)  # torch.Size([32, 2])
        # print(y.shape)

        return y

    def size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)