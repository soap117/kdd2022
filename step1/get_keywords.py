import pickle
import torch
import numpy as np
from tqdm import tqdm
import re
from transformers import BertTokenizer
from transformers import BertModel
from keras.preprocessing.sequence import pad_sequences
import os
from pprint import pprint

CLS = '[CLS]'
SEP = '[SEP]'

bert_model = 'hfl/chinese-bert-wwm-ext'
tokenizer = BertTokenizer.from_pretrained(bert_model)
model = BertModel.from_pretrained(bert_model)
keywords_all = {}

def tokenize(txt):
    src = re.sub('\*\*', '', txt).lower()
    tokens = tokenizer.tokenize(src)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    return ids

def preprocess(path_from):
    with open(os.path.join(path_from,'dataset-aligned.pkl'), 'rb') as f:
        dataset_aligned = pickle.load(f)

    keywords = []
    for data in tqdm(dataset_aligned):
        content = data[2]
        for cont in content:
            for tooltip in cont['tooltips']:
                keyword = tooltip['origin']
                if keyword in ['痰','肽','T','铝','痹','脾','醛','氨','铁','T','P','R',
                               'M','硒','痹','科','氡','为','肽','但','瘀','痰','锰','U','数','0','酚','研','肺','']:
                    # print(keyword)
                    continue
                keywords.append(keyword)
                inputs = tokenizer(keyword,return_tensors="pt")
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state
                embedding = embeddings.squeeze().mean(dim=0)
                keywords_all[tooltip['origin']] = embedding


def main():
    print('train dataset:')
    preprocess('../../data/train')
    print('test dataset:')
    preprocess('../../data/test')
    print('valid dataset:')
    preprocess('../../data/valid')
    with open('./data/keywords.pkl','wb') as f:
        pickle.dump(keywords_all, f)
    print(len(keywords_all))

if __name__ == '__main__':
    main()



'''
TPRM硒痹科氡为肽但瘀痰锰U数0酚研肺
'''