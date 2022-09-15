# -*- coding:utf-8 -*-

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime
bar_format = '{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'

from Core.Dataset import Dataset
from utils import get_negative_pair

from Core.Model import TargetFM1
from Core.Model import TargetFM2

def predict_result(mirna_fasta_file, mrna_fasta_file, query_file, model=None, weight_file=None,
                   seed_match='offset-9-mer-m7', level='gene', batch_size=32, output_file=None, device='cpu'):
    """
    if not isinstance(model, deepTarget):
        raise ValueError("'model' expected <nn.Module 'deepTarget'>, got {}".format(type(model)))
    """

    if not weight_file.endswith('.pt'):
        raise ValueError("'weight_file' expected '*.pt', got {}".format(weight_file))

    model.load_state_dict(torch.load(weight_file))

    test_set = Dataset(mirna_fasta_file, mrna_fasta_file, query_file, seed_match=seed_match, header=True, train=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    y_probs = []
    y_predicts = []
    y_truth = []

    model = model.to(device)
    with torch.no_grad():
        model.eval()

        with tqdm(test_loader, bar_format=bar_format) as tqdm_loader:
            for i, ((mirna, mrna), label) in enumerate(tqdm_loader):
                mirna, mrna, label = mirna.to(device, dtype=torch.int64), mrna.to(device, dtype=torch.int64), label.to(
                    device)

                outputs = model(mirna, mrna)
                _, predicts = torch.max(outputs.data, 1)
                probabilities = F.softmax(outputs, dim=1)

                y_probs.extend(probabilities.cpu().numpy()[:, 1])
                y_predicts.extend(predicts.cpu().numpy())
                y_truth.extend(label.cpu().numpy())

                global correct
                # print(predicts.cpu().numpy())
                # print(label.cpu().numpy())
                correct += (predicts == label).sum().item()
        print(len(test_set))
        print("acc:", float(correct / len(test_set)) * 100, "%")

        if output_file is None:
            time = datetime.now()
            output_file = "{}.csv".format(time.strftime('%Y%m%d_%H%M%S_results'))
        results = postprocess_result(test_set.dataset, y_probs, y_predicts,
                                     seed_match=seed_match, level=level, output_file=output_file)

        print(results)


""" 结果统计 """
def postprocess_result(dataset, probabilities, predicts, predict_mode=True, output_file=None, cts_size=30, seed_match='offset-9-mer-m7', level='site'):
    neg_pairs = get_negative_pair(dataset['mirna_fasta_file'], dataset['mrna_fasta_file'], dataset['ground_truth_file'], cts_size=cts_size, seed_match=seed_match, predict_mode=predict_mode)   # 负样本对

    # dataset:正样本集  neg_pair:负样本集
    query_ids = np.append(dataset['query_ids'], neg_pairs['query_ids'])
    target_ids = np.append(dataset['target_ids'], neg_pairs['target_ids'])
    target_locs = np.append(dataset['target_locs'], neg_pairs['target_locs'])
    probabilities = np.append(probabilities, neg_pairs['probabilities'])        # probabilities：正样本训练经模型得到的prob  neg_pairs['probabilities]：构造负样本时设定好的prob=0.0
    predicts = np.append(predicts, neg_pairs['predicts'])                       # predicts:正样本训练经模型得到的预测   neg_pairs['predicts']：构造负样本时设定好的prob=-1

    # output format: [QUERY, TARGET, LOCATION, PROBABILITY]
    records = pd.DataFrame(columns=['MIRNA_ID', 'MRNA_ID', 'LOCATION', 'PROBABILITY'])
    records['MIRNA_ID'] = query_ids
    records['MRNA_ID'] = target_ids
    records['LOCATION'] = np.array(["{},{}".format(max(1, l-cts_size+1), l) if l != -1 else "-1,-1" for l in target_locs])
    records['PROBABILITY'] = probabilities
    if predict_mode is True:                  # 是否在预测
        records['PREDICT'] = predicts
    else:
        records['LABEL'] = predicts

    # site level
    records = records.sort_values(by=['PROBABILITY', 'MIRNA_ID', 'MRNA_ID'], ascending=[False, True, True])  # sort_values()函数原理类似于SQL中的order by，将数据集依照某个字段中的数据进行排序； ascending	是否按指定列的数组升序排列，默认为True，即升序排列
    # gene level
    unique_records = records.sort_values(by=['PROBABILITY', 'MIRNA_ID', 'MRNA_ID'], ascending=[False, True, True]).drop_duplicates(subset=['MIRNA_ID', 'MRNA_ID'], keep='first')

    if level == 'site':
        if output_file is not None:
            records.to_csv(output_file, index=False, sep='\t')
        return records

    elif level == 'gene':
        if output_file is not None:
            unique_records.to_csv(output_file, index=False, sep='\t')
        return unique_records

    else:
        raise ValueError("level expected 'site' or 'gene', got '{}'".format(mode))


start_time = datetime.now()
print("\n[START] {}".format(start_time.strftime('%Y-%m-%d @ %H:%M:%S')))

mirna_fasta_file = 'Data/mirna.fasta'
mrna_fasta_file = 'Data/mrna.fasta'
query_file = 'Data/Test/test_split_0.csv'
model1 = TargetFM1()
model2 = TargetFM2()
weight_file = '/content/drive/MyDrive/Bio/RRIs/TargetFM/weights.pt'
seed_match = 'offset-9-mer-m7'
level = 'gene'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

results = predict_result(mirna_fasta_file, mrna_fasta_file, query_file,
                         model=model1, weight_file=weight_file,
                         seed_match=seed_match, level=level,
                         output_file=None, device=device)

finish_time = datetime.now()
print("\n[FINISH] {} (user time: {})\n".format(finish_time.now().strftime('%Y-%m-%d @ %H:%M:%S'),
                                               (finish_time - start_time)))