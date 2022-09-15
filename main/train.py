# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from tqdm.auto import tqdm
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime

from Core.Dataset import TrainDataset
from Core.Dataset import Dataset
from Core.Model import TargetFM1
from Core.Model import TargetFM2

bar_format = '{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'

def train_model(mirna_fasta_file, mrna_fasta_file, train_file, model=None, cts_size=30, seed_match='offset-9-mer-m7',
                level='gene', batch_size=32, epochs=10, save_file=None, device='cpu'):
    """
    if not isinstance(model, deepTarget):
        raise ValueError("'model' expected <nn.Module 'deepTarget'>, got {}".format(type(model)))

    print("\n[TRAIN] {}".format(model.name))
    """

    if train_file.split('/')[-1] == 'train_set.csv':
        train_set = TrainDataset(train_file)
    else:
        # 实例化
        train_set = Dataset(mirna_fasta_file, mrna_fasta_file, train_file, seed_match=seed_match, header=True,
                            train=True)  # return (mirna, mrna), label
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

    class_weight = torch.Tensor(
        compute_class_weight('balanced', classes=np.unique(train_set.labels), y=train_set.labels)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = optim.Adam(model.parameters())

    model = model.to(device)
    for epoch in range(epochs):
        epoch_loss, corrects = 0, 0

        with tqdm(train_loader, desc="Epoch {}/{}".format(epoch + 1, epochs), bar_format=bar_format) as tqdm_loader:
            for i, ((mirna, mrna), label) in enumerate(tqdm_loader):

                mirna, mrna, label = mirna.to(device, dtype=torch.int64), mrna.to(device, dtype=torch.int64), label.to(
                    device)

                outputs = model(mirna, mrna)
                loss = criterion(outputs, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * outputs.size(0)
                corrects += (torch.max(outputs, 1)[1] == label).sum().item()

                if (i + 1) == len(train_loader):
                    tqdm_loader.set_postfix(dict(loss=(epoch_loss / len(train_set)), acc=(corrects / len(train_set))))
                else:
                    tqdm_loader.set_postfix(loss=loss.item())

    if save_file is None:
        time = datetime.now()
        save_file = "{}.pt".format(time.strftime('%Y%m%d_%H%M%S_weights'))
    torch.save(model.state_dict(), save_file)


start_time = datetime.now()
print("\n[START] {}".format(start_time.strftime('%Y-%m-%d @ %H:%M:%S')))

mirna_fasta_file = 'Data/mirna.fasta'
mrna_fasta_file = 'Data/mrna.fasta'
seed_match = 'offset-9-mer-m7'
level = 'gene'
train_file = 'Data/Train/train_set.csv'
weight_file = 'weights.pt'
batch_size = 32
epochs = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model1 = TargetFM1()
model2 = TargetFM2()

torch.save(model1.state_dict(), weight_file)
train_model(mirna_fasta_file, mrna_fasta_file, train_file,
            model=model1,
            seed_match=seed_match, level=level,
            batch_size=batch_size, epochs=epochs,
            save_file=weight_file, device=device)

finish_time = datetime.now()
print("\n[FINISH] {} (user time: {})\n".format(finish_time.now().strftime('%Y-%m-%d @ %H:%M:%S'), (finish_time - start_time)))