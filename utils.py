# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Alphabet import generic_rna
import regex

""" 确定mRNA上的CTS片段位置 """
def find_candidate(mirna_sequence, mrna_sequence, seed_match):
    positions = set()

    # 确定seed_match方式
    if seed_match == '10-mer-m6':
        SEED_START = 1
        SEED_END = 10
        SEED_OFFSET = SEED_START - 1
        MIN_MATCH = 6
        TOLERANCE = (SEED_END - SEED_START + 1) - MIN_MATCH
    elif seed_match == '10-mer-m7':
        SEED_START = 1
        SEED_END = 10
        SEED_OFFSET = SEED_START - 1
        MIN_MATCH = 7
        TOLERANCE = (SEED_END - SEED_START + 1) - MIN_MATCH
    elif seed_match == 'offset-9-mer-m7':
        SEED_START = 2
        SEED_END = 10
        SEED_OFFSET = SEED_START - 1
        MIN_MATCH = 7
        TOLERANCE = (SEED_END - SEED_START + 1) - MIN_MATCH
    elif seed_match == 'strict':
        positions = find_strict_candidate(mirna_sequence, mrna_sequence)
        return positions

    else:
        raise ValueError(
            "seed_match expected 'strict', '10-mer-m6', '10-mer-m7', or 'offset-9-mer-m7', got '{}'".format(seed_match))

    # 确定mirna上对应seed区域
    seed = mirna_sequence[(SEED_START - 1):SEED_END]

    # complement()返回序列的转录序列； rc_seed:seed的配对片段
    rc_seed = str(Seq(seed, generic_rna).complement())

    # 在mrna中找可以与seed匹配的片段； re.finditer(pattern, string, flags=0), Use the finditer() function to match a pattern in a string and return an iterator yielding the Match objects.
    match_iter = regex.finditer("({}){{e<={}}}".format(rc_seed, TOLERANCE), mrna_sequence)

    for match_index in match_iter:
        # positions.add(match_index.start()) # slice-start indicies
        positions.add(match_index.end() + SEED_OFFSET)  # slice-stop indicies

    # CTS片段的第一个token位置
    positions = list(positions)

    return positions


def find_strict_candidate(mirna_sequence, mrna_sequence):
    positions = set()

    SEED_TYPES = ['8-mer', '7-mer-m8', '7-mer-A1', '6-mer', '6-mer-A1', 'offset-7-mer', 'offset-6-mer']
    for seed_match in SEED_TYPES:
        if seed_match == '8-mer':
            SEED_START = 2
            SEED_END = 8
            SEED_OFFSET = 0
            seed = 'U' + mirna_sequence[(SEED_START - 1):SEED_END]
        elif seed_match == '7-mer-m8':
            SEED_START = 1
            SEED_END = 8
            SEED_OFFSET = 0
            seed = mirna_sequence[(SEED_START - 1):SEED_END]
        elif seed_match == '7-mer-A1':
            SEED_START = 2
            SEED_END = 7
            SEED_OFFSET = 0
            seed = 'U' + mirna_sequence[(SEED_START - 1):SEED_END]
        elif seed_match == '6-mer':
            SEED_START = 2
            SEED_END = 7
            SEED_OFFSET = 1
            seed = mirna_sequence[(SEED_START - 1):SEED_END]
        elif seed_match == '6mer-A1':
            SEED_START = 2
            SEED_END = 6
            SEED_OFFSET = 0
            seed = 'U' + mirna_sequence[(SEED_START - 1):SEED_END]
        elif seed_match == 'offset-7-mer':
            SEED_START = 3
            SEED_END = 9
            SEED_OFFSET = 0
            seed = mirna_sequence[(SEED_START - 1):SEED_END]
        elif seed_match == 'offset-6-mer':
            SEED_START = 3
            SEED_END = 8
            SEED_OFFSET = 0
            seed = mirna_sequence[(SEED_START - 1):SEED_END]

        rc_seed = str(Seq(seed, generic_rna).complement())
        match_iter = regex.finditer(rc_seed, mrna_sequence)

        for match_index in match_iter:
            # positions.add(match_index.start()) # slice-start indicies
            positions.add(match_index.end() + SEED_OFFSET)  # slice-stop indicies

    positions = list(positions)

    return positions


""" 确定CTS片段 """
def get_candidate(mirna_sequence, mrna_sequence, cts_size, seed_match):
    positions = find_candidate(mirna_sequence, mrna_sequence, seed_match)  # CTS片段首位置（list）

    candidates = []
    for i in positions:
        site_sequence = mrna_sequence[max(0, i - cts_size):i]  # mrna上CTS片段
        rev_site_sequence = site_sequence[::-1]  # [::-1]序列翻转：从左到右->从右到左
        rc_site_sequence = str(Seq(rev_site_sequence, generic_rna).complement())  # 转录序列
        candidates.append(rev_site_sequence)  # miRNAs: 5'-ends to 3'-ends,  mRNAs: 3'-ends to 5'-ends
        # candidates.append(rc_site_sequence)

    return candidates, positions


""" 生成pairs """
def make_pair(mirna_sequence, mrna_sequence, cts_size, seed_match):
    candidates, positions = get_candidate(mirna_sequence, mrna_sequence, cts_size, seed_match)  # candidates:list of seq

    mirna_querys = []  # mirna seed region
    mrna_targets = []  # mirna CTS
    if len(candidates) == 0:  # mrna中无mirna匹配片段
        return (mirna_querys, mrna_targets, positions)
    else:
        mirna_sequence = mirna_sequence[0:cts_size]  # mirna的seed region
        for i in range(len(candidates)):
            mirna_querys.append(mirna_sequence)
            mrna_targets.append(candidates[i])  # 一个mirna的seed对应多个mrna的CTS

    return mirna_querys, mrna_targets, positions


""" 读取mirna,mrna序列 """
def read_fasta(mirna_fasta_file, mrna_fasta_file):
    mirna_list = list(SeqIO.parse(mirna_fasta_file, 'fasta'))
    mrna_list = list(SeqIO.parse(mrna_fasta_file, 'fasta'))

    mirna_ids = []
    mirna_seqs = []
    mrna_ids = []
    mrna_seqs = []

    for i in range(len(mirna_list)):
        mirna_ids.append(str(mirna_list[i].id))
        mirna_seqs.append(str(mirna_list[i].seq))

    for i in range(len(mrna_list)):
        mrna_ids.append(str(mrna_list[i].id))
        mrna_seqs.append(str(mrna_list[i].seq))

    return mirna_ids, mirna_seqs, mrna_ids, mrna_seqs


""" 读取gt(label) """
def read_ground_truth(ground_truth_file, header=True, train=False):
    # input format: [MIRNA_ID, MRNA_ID, LABEL]
    if header is True:
        records = pd.read_csv(ground_truth_file, header=0, sep='\t')
    else:
        records = pd.read_csv(ground_truth_file, header=None, sep='\t')

    query_ids = np.asarray(records.iloc[:, 0].values)
    target_ids = np.asarray(records.iloc[:, 1].values)
    if train is True:
        labels = np.asarray(records.iloc[:, 2].values)
    else:
        labels = np.full((len(records),), fill_value=-1)

    return query_ids, target_ids, labels


""" 核苷酸转整型 """
# AUUCAAU -> 1442114
def nucleotide_to_int(nucleotides, max_len):
    dictionary = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'U': 4}

    chars = []
    nucleotides = nucleotides.upper()  # nucleotides小写转大写
    for c in nucleotides:
        chars.append(c)

    ints_enc = np.full((max_len,),
                       fill_value=0)  # to post-pad inputs; np.full(shape, fill_value)返回一个给定大小和类型并且以指定数字全部填充的新数组
    for i in range(len(chars)):
        try:
            ints_enc[i] = dictionary[chars[i]]
        except KeyError:
            continue
        except IndexError:
            break

    return ints_enc


""" 序列转整型 """
# 所有序列转整型
def sequence_to_int(sequences, max_len):
    import itertools

    if type(sequences) is list:
        seqs_enc = np.asarray([nucleotide_to_int(seq, max_len) for seq in sequences])
    else:
        seqs_enc = np.asarray([nucleotide_to_int(seq, max_len) for seq in sequences])
        seqs_enc = list(itertools.chain(*seqs_enc))
        seqs_enc = np.asarray(seqs_enc)

    return seqs_enc


""" 统一序列长度，两种pad方式 """
def pad_sequences(sequences, max_len=None, padding='pre', fill_value='O'):
    n_samples = len(sequences)  # 样本数：序列个数； sequences:list of sequence

    lengths = []
    for seq in sequences:
        try:
            lengths.append(len(seq))  # 记录每一个序列的长度
        except TypeError:
            raise ValueError("sequences expected a list of iterables, got {}".format(seq))
    if max_len is None:
        max_len = np.max(lengths)  # 确定最大序列长度

    # input_shape = np.asarray(sequences[0]).shape[1:]   # ???
    # padded_shape = (n_samples, max_len) + input_shape
    # padded = np.full(padded_shape, fill_value=fill_value)

    # import pdb; pdb.set_trace()

    for i, seq in enumerate(sequences):
        if padding == 'pre':
            if max_len > len(seq):
                sequences[i] = [fill_value] * (max_len - len(seq)) + sequences[i]
            else:
                sequences[i] = sequences[i][:max_len]
        elif padding == 'post':
            if max_len > len(seq):
                sequences[i] = sequences[i] + [fill_value] * (max_len - len(seq))
            else:
                sequences[i] = sequences[i][:max_len]
        else:
            raise ValueError("padding expected 'pre' or 'post', got {}".format(truncating))

    return sequences


""" 对label进行编码 （samples, classes）"""
def to_categorical(labels, n_classes=None):
    labels = np.array(labels, dtype='int').reshape(-1)

    n_samples = labels.shape[0]
    if not n_classes:
        n_classes = np.max(labels) + 1

    categorical = np.zeros((n_samples, n_classes))
    categorical[np.arange(n_samples), labels] = 1

    return categorical


""" 对miran,mrna,y进行one-hot编码 """
def preprocess_data(x_query_seqs, x_target_seqs, y=None, cts_size=None, pre_padding=False):
    if cts_size is not None:
        max_len = cts_size
    else:
        max_len = max(len(max(x_query_seqs, key=len)), len(max(x_target_seqs, key=len)))

    # 将mirna,mran转为整型
    # x_mirna = sequence_to_int(x_query_seqs, max_len)
    # x_mrna = sequence_to_int(x_target_seqs, max_len)

    # padding, max取max(mirna,mrna)
    # if pre_padding:
    x_query_seqs = [list(i) for i in x_query_seqs]
    x_target_seqs = [list(i) for i in x_target_seqs]
    x_mirna = pad_sequences(x_query_seqs, max_len, padding='pre')
    x_mrna = pad_sequences(x_target_seqs, max_len, padding='pre')

    # 对mrna,mirna进行one-hot编码
    # x_mirna_embd = one_hot_enc(x_mirna)
    # x_mrna_embd = one_hot_enc(x_mrna)
    if y is not None:
        y_embd = to_categorical(y, np.unique(y).size)

        return x_mirna, x_mrna, y_embd
    else:
        return x_mirna, x_mrna

""" 构造dataset(字典) """
def make_input_pair(mirna_fasta_file, mrna_fasta_file, ground_truth_file, cts_size=30, seed_match='offset-9-mer-m7', header=True, train=True):
    mirna_ids, mirna_seqs, mrna_ids, mrna_seqs = read_fasta(mirna_fasta_file, mrna_fasta_file)           # mirna_ids, mirna_seqs, mrna_ids, mrna_seqs
    query_ids, target_ids, labels = read_ground_truth(ground_truth_file, header=header, train=train)     # query_ids, target_ids, labels

    dataset = {
        'mirna_fasta_file': mirna_fasta_file,
        'mrna_fasta_file': mrna_fasta_file,
        'ground_truth_file': ground_truth_file,
        'query_ids': [],
        'query_seqs': [],
        'target_ids': [],
        'target_seqs': [],
        'target_locs': [],
        'labels': []
    }

    for i in range(len(query_ids)):
        try:
            j = mirna_ids.index(query_ids[i])    # j:mirna index
        except ValueError:
            continue
        try:
            k = mrna_ids.index(target_ids[i])    # k:mrna index
        except ValueError:
            continue

        query_seqs, target_seqs, locations = make_pair(mirna_seqs[j], mrna_seqs[k], cts_size=cts_size, seed_match=seed_match)

        n_pairs = len(locations)    # 产生的 mirna-mrna匹配对数
        if n_pairs > 0:
            queries = [query_ids[i] for n in range(n_pairs)]
            dataset['query_ids'].extend(queries)
            dataset['query_seqs'].extend(query_seqs)

            targets = [target_ids[i] for n in range(n_pairs)]
            dataset['target_ids'].extend(targets)
            dataset['target_seqs'].extend(target_seqs)
            dataset['target_locs'].extend(locations)

            dataset['labels'].extend([[labels[i]] for p in range(n_pairs)])

    return dataset

""" 无labels """
def make_brute_force_pair(mirna_fasta_file, mrna_fasta_file, cts_size=30, seed_match='offset-9-mer-m7'):
    mirna_ids, mirna_seqs, mrna_ids, mrna_seqs = read_fasta(mirna_fasta_file, mrna_fasta_file)

    dataset = {
        'query_ids': [],
        'query_seqs': [],
        'target_ids': [],
        'target_seqs': [],
        'target_locs': []
    }

    for i in range(len(mirna_ids)):
        for j in range(len(mrna_ids)):
            query_seqs, target_seqs, positions = make_pair(mirna_seqs[i], mrna_seqs[j], cts_size, seed_match)

            n_pairs = len(positions)
            if n_pairs > 0:
                query_ids = [mirna_ids[i] for k in range(n_pairs)]
                dataset['query_ids'].extend(query_ids)
                dataset['query_seqs'].extend(query_seqs)

                target_ids = [mrna_ids[j] for k in range(n_pairs)]
                dataset['target_ids'].extend(target_ids)
                dataset['target_seqs'].extend(target_seqs)
                dataset['target_locs'].extend(positions)

    return dataset


""" 生成负样本 """


def get_negative_pair(mirna_fasta_file, mrna_fasta_file, ground_truth_file=None, cts_size=30,
                      seed_match='offset-9-mer-m7', header=False, predict_mode=True):
    mirna_ids, mirna_seqs, mrna_ids, mrna_seqs = read_fasta(mirna_fasta_file, mrna_fasta_file)  # 读取mrna,mirna文件

    dataset = {
        'query_ids': [],
        'target_ids': [],
        'predicts': []
    }

    if ground_truth_file is not None:
        query_ids, target_ids, labels = read_ground_truth(ground_truth_file, header=header)

        for i in range(len(query_ids)):
            try:
                j = mirna_ids.index(query_ids[i])
            except ValueError:
                continue
            try:
                k = mrna_ids.index(target_ids[i])
            except ValueError:
                continue

            query_seqs, target_seqs, locations = make_pair(mirna_seqs[j], mrna_seqs[k], cts_size=cts_size,
                                                           seed_match=seed_match)

            n_pairs = len(locations)
            if (n_pairs == 0) and (predict_mode is True):
                dataset['query_ids'].append(query_ids[i])
                dataset['target_ids'].append(target_ids[i])
                dataset['predicts'].append(0)
            elif (n_pairs == 0) and (predict_mode is False):
                dataset['query_ids'].append(query_ids[i])
                dataset['target_ids'].append(target_ids[i])
                dataset['predicts'].append(labels[i])
    else:
        for i in range(len(mirna_ids)):
            for j in range(len(mrna_ids)):
                query_seqs, target_seqs, locations = make_pair(mirna_seqs[i], mrna_seqs[j], cts_size=cts_size,
                                                               seed_match=seed_match)

                n_pairs = len(locations)
                if n_pairs == 0:
                    dataset['query_ids'].append(mirna_ids[i])
                    dataset['target_ids'].append(mrna_ids[j])
                    dataset['predicts'].append(0)

    dataset['target_locs'] = [-1 for i in range(len(dataset['query_ids']))]
    dataset['probabilities'] = [0.0 for i in range(len(dataset['query_ids']))]

    return dataset