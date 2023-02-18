import json

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
labels = {}
for token in tokenizer.vocab:
    labels[token] = 0

num_lines = 0
num_tokens = 0
paths = ['./books_large_p1.txt', './books_large_p2.txt']
for path in paths:
    with open(path, mode='r') as f:
        while True:
            line = f.readline()
            if line == '':
                break
            sentence_list = tokenizer.tokenize(line)
            for token in sentence_list:
                labels[token] += 1
                num_tokens += 1
            num_lines += 1
            if num_lines % 10000 == 0:
                print(str(num_lines) + ' lines have been executed.')
    f.close()
print(str(num_tokens) + ' tokens in this Corpus!')

labels_sort = sorted(labels.items(), key=lambda x: x[1], reverse=True)

with open('labels.json', mode='w') as f:
    json.dump(labels, f, indent=2)

with open('labels_sort.json', mode='w') as f:
    json.dump(labels_sort, f, indent=2)
