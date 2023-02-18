import torch
import pandas as pd
from transformers import BertModel, BertTokenizer

# model_path = "./output/unsup-consert-base-07260900/0_Transformer/"  # baseline 32
# model_path = "./output/unsup-consert-base-07261600/0_Transformer/"  # adversarial
model_path = "./output/unsup-consert-base-07271600/0_Transformer/"  # ours
# model_path = 'bert-base-uncased'

DEVICE = 'cuda:3'
model = BertModel.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
example = 'a man is playing a bamboo flute.'
inputs = tokenizer(example, return_tensors='pt')
outputs = model(**inputs, output_attentions=True)
att = outputs[-1][-1][:, -3, :, :].squeeze()
data = pd.DataFrame(att.detach().numpy())
writer = pd.ExcelWriter('attentions.xlsx')
data.to_excel(writer, 'page_0', float_format='%.5f')
writer.save()
writer.close()

