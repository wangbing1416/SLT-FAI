# SLT- FAI
This repo is the released code of our work *SLT-FAI: Unsupervised Sentence Representation Learning with Frequency-induced Adversarial Tuning and Incomplete Sentence Filtering*

Our released code follows to ConSERT: [ConSERT: A Contrastive Framework for Self-Supervised Sentence Representation Transfer](https://aclanthology.org/2021.acl-long.393/)

### Requirements

```
torch==1.6.0
cudatoolkit==10.0.103
cudnn==7.6.5
sentence-transformers==0.3.9
transformers==3.4.0
tensorboardX==2.1
pandas==1.1.5
sentencepiece==0.1.85
matplotlib==3.4.1
apex==0.1.0
```

To install apex, run:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./
```



### Train

- Download pre-trained language model or `BERTModel.from_pretrained`.

- Download STS datasets to `./data` folder by running `cd data && bash get_transfer_data.bash`.

- Run the scripts in the folder `./scripts` to reproduce our experiments `bash scripts/unsup-SLTFAI-base.sh`:



### Evaluation

- **self-attention weight**

```python
python ./testAtt.py
```

- **embedding visualization**, its outputs are an EXCEL file

```python
python ./testGRL.py
```

- **uniformity and alignment**

```python
python ./testUA.py
```



### Citation
None