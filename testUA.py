import torch
import tqdm
import os
import time
import argparse
from torch.utils.data import DataLoader
from torch import nn, Tensor
from torch.nn.functional import normalize
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers import models, losses
from data_utils import load_datasets
from sentence_transformers.util import batch_to_device
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional, Set

parser = argparse.ArgumentParser()
# model_path = 'bert-base-uncased'
# model_path = "./output/unsup-consert-base-06281629/0_Transformer/"  # consert

# model_path = "./output/unsup-consert-base-07260900/0_Transformer/"  # baseline 32
# model_path = "./output/unsup-consert-base-07261600/0_Transformer/"  # adversarial
# model_path = "./output/unsup-consert-base-07271600/0_Transformer/"  # ours
parser.add_argument("--model", default='bert-base-uncased')
parser.add_argument("--dataset", default="stsb", help="sts12, sts13, sts14, sts15, sts16, stsb, sickr")
parser.add_argument("--device", default='cuda:3')
parser.add_argument("--p", type=int, default=2)
parser.add_argument("--root", default='au-result')
args = parser.parse_args()
model_path = args.model
# model_path = "./output/unsup-consert-base-07092000/0_Transformer/"
tokenizer = BertTokenizer.from_pretrained(model_path)
device = args.device


def _recover_to_origin_keys(sentence_feature: Dict[str, Tensor], ori_keys: Set[str]):
    return {k: v for k, v in sentence_feature.items() if k in ori_keys}


def _data_aug(model, sentence_feature, name, ori_keys, cutoff_rate=0.2):
    assert name in ("none", "shuffle", "token_cutoff", "feature_cutoff", "dropout", "span")
    sentence_feature = _recover_to_origin_keys(sentence_feature, ori_keys)
    if name == "none":
        pass  # do nothing
    elif name == "shuffle":
        model[0].auto_model.set_flag("data_aug_shuffle", True)
    elif name == "token_cutoff":
        model[0].auto_model.set_flag("data_aug_cutoff", True)
        model[0].auto_model.set_flag("data_aug_cutoff.direction", "row")
        model[0].auto_model.set_flag("data_aug_cutoff.rate", cutoff_rate)
    elif name == "span":
        model[0].auto_model.set_flag("data_aug_span", True)
        model[0].auto_model.set_flag("data_aug_span.rate", cutoff_rate)
    elif name == "feature_cutoff":
        model[0].auto_model.set_flag("data_aug_cutoff", True)
        model[0].auto_model.set_flag("data_aug_cutoff.direction", "column")
        model[0].auto_model.set_flag("data_aug_cutoff.rate", cutoff_rate)
    elif name == "dropout":
        model[0].auto_model.set_flag("data_aug_cutoff", True)
        model[0].auto_model.set_flag("data_aug_cutoff.direction", "random")
        model[0].auto_model.set_flag("data_aug_cutoff.rate", cutoff_rate)
    rep = model(sentence_feature)["sentence_embedding"]
    return rep, sentence_feature['token_embeddings']


train_samples = load_datasets(datasets=[args.dataset],
                              need_label=False, use_all_unsupervised_texts=True, no_pair=True)
word_embedding_model = models.Transformer(model_path)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True, pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model]).to(device)
train_dataset = SentencesDataset(train_samples, model=model)
train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=1)
train_dataloader.collate_fn = model.smart_batching_collate

align = []
all_rep = []
for data in tqdm.tqdm(train_dataloader):
    features, labels = batch_to_device(data, device)
    sentence_feature_a = features[0]
    ori_feature_keys = set(sentence_feature_a.keys())  # record the keys since the features will be updated
    rep, _ = _data_aug(model, sentence_feature_a, 'none', ori_feature_keys)
    rep_a_view1, _ = _data_aug(model, sentence_feature_a, 'feature_cutoff', ori_feature_keys, 0.2)
    rep_a_view2, _ = _data_aug(model, sentence_feature_a, 'shuffle', ori_feature_keys)
    rep, rep_a_view1, rep_a_view2 = normalize(rep), normalize(rep_a_view1), normalize(rep_a_view2)
    align.append(torch.norm(rep_a_view1 - rep_a_view2, p=args.p).item() ** 2)
    all_rep.append(rep.detach().to('cpu'))

uniform = []
for i in tqdm.tqdm(range(len(all_rep))):
    for j in range(i + 1, len(all_rep)):
        uniform.append(torch.exp(torch.norm(all_rep[i] - all_rep[j], p=args.p) ** 2 * (-2)))

alignment = sum(align) / len(align)
uniformv = torch.log(sum(uniform) / len(uniform))
print('sample scale is {}'.format(len(align)))
print('alignment: {:.4f}'.format(alignment))
print('uniform: {:.4f}'.format(uniformv))

if not os.path.exists(args.root):
    os.makedirs(args.root)
path = os.path.join(args.root, '{}-{}.txt'.format(args.dataset, time.asctime()))
with open(path, 'w') as f:
    f.write('dataset: {}, '.format(args.dataset))
    f.write('model: {}, '.format(args.model))
    f.write('sample scale is {}, '.format(len(align)))
    f.write('alignment: {:.4f}, '.format(alignment))
    f.write('uniform: {:.4f}'.format(uniformv))
f.close()
