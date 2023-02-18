from torch.autograd import Function
from typing import Any, Optional, Tuple
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import json
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA, TruncatedSVD
from transformers import BertModel, BertTokenizer

from data_utils import load_datasets

path = './data/labels/bookcorpus/labels.json'


# model_path = "./output/unsup-consert-base-06281629/0_Transformer/"
# model_path = "./output/unsup-consert-base-07092000/0_Transformer/"  # uniform
# model_path = "./output/unsup-consert-base-07091600/0_Transformer/"
# model_path = 'bert-base-uncased'

# model_path = "./output/unsup-consert-base-07260900/0_Transformer/"  # baseline 32
# model_path = "./output/unsup-consert-base-07261600/0_Transformer/"  # adversarial
model_path = "./output/unsup-consert-base-07271600/0_Transformer/"  # ours
low_rate = 0.5
tokenizer = BertTokenizer.from_pretrained(model_path)

device = 'cuda:2'
with open(path, 'r') as f:
        token_dic = json.load(f)
        freq_list = [token_dic[i] for i in token_dic]
        num = 0
        for i in freq_list:
            if i == 0:
                num += 1
        freq_list.sort()
        thres = freq_list[num + int((len(freq_list) - num) * low_rate)]
        index_dic = {}
        freq_label = {}
        for k, v in token_dic.items():
            index = tokenizer.convert_tokens_to_ids(k)
            index_dic[index] = v
            freq_label[index] = 1 if v < thres else 0

# tsne = TSNE(n_components=2, init='pca', verbose=1)
pca = TruncatedSVD(n_components=2)

train_samples = load_datasets(datasets=["sts12", "sts13", "sts14", "sts15", "sts16", "stsb", "sickr"],
                              need_label=False, use_all_unsupervised_texts=True)
model = BertModel.from_pretrained(model_path).to(device)

res = []
label_0 = []
label_1 = []
i = 0
with torch.no_grad():
    for sample in tqdm(train_samples[:10000]):
        sample = sample.texts[0]
        sentence = tokenizer(sample, return_tensors='pt').to(device)
        for index in sentence['input_ids'][0]:
            if freq_label[index.item()] == 0:
                label_0.append(i)
            else:
                label_1.append(i)
            i += 1

        # emb = model(**sentence)[0][:, 0, :]

        emb = torch.mean(model(**sentence)[0], dim=1)
        # emb = model(**sentence)[0].squeeze()
        res.append(emb)
res = torch.cat(res, dim=0).to('cpu')
temp = pca.fit_transform(res.numpy())

# data_0 = pd.DataFrame(temp[label_0])
# data_1 = pd.DataFrame(temp[label_1])
data = pd.DataFrame(temp)

writer = pd.ExcelWriter('embedding.xlsx')		# 写入Excel文件

# data_0.to_excel(writer, 'page_0', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
# data_1.to_excel(writer, 'page_1', float_format='%.5f')
data.to_excel(writer, 'page_0', float_format='%.5f')
writer.save()

writer.close()

