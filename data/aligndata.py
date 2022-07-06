import os.path
import pickle
import torch
from tqdm import tqdm
import re
import random
import os
import json
from align import creat_sentence

def aligneddata(dataset,path):
    # with open(os.path.join(path,'dataset.pkl'),'rb') as f:
    #     dataset = pickle.load(f)
    # src_all = [u[0] for u in dataset]
    # tar_all = [u[1] for u in dataset]
    src_all = [u['src'] for u in dataset]
    tar_all = [u['tar'] for u in dataset]
    content_all = [u['contents'] for u in dataset]

    for i in range(len(src_all)):
        src = src_all[i]
        src = re.sub('\*\*', '', src).lower()
        src = src.replace('\n', '').replace('。。', '。')
        src_all[i] = src

    for i in range(len(tar_all)):
        tar = tar_all[i]
        tar = re.sub('\*\*', '', tar).lower()
        tar = tar.replace('\n', '').replace('。。', '。')
        tar_all[i] = tar

    print(len(dataset))

    dataset_new = []
    dataset_new_para = []
    dataset = []

    for src, tar, contents in zip(src_all, tar_all, content_all):
        if len(src)==0 or len(tar)==0: continue
        # if src[-1] == '。' and tar[-1] != '。':
        #     tar += '。'
        # if tar[-1] == '。' and src[-1] != '。':
        #     src += '。'
        # srcs = src.strip('。').split('。')
        # tars = tar.strip('。').split('。')
        srcs, tars = creat_sentence(src, tar)
        if len(srcs) == len(tars):
            dataset.append((src,tar))
            for u,v in zip(srcs, tars):
                if len(u)>2 and len(v)>2:
                    dataset_new.append((u, v))
        querys = []
        for content in contents:
            for tooltip in content['tooltips']:
                querys.append((tooltip['origin'], content['text'], tooltip['translation']))

        tars_sec = '。'.join(tars)
        srcs_sec = '。'.join(srcs)
        if len(srcs_sec) > 2 and len(tars_sec) > 2:
            dataset_new_para.append((srcs_sec, tars_sec, querys))

    print(len(dataset))
    with open(os.path.join(path, 'dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)
    print(len(dataset_new))
    with open(os.path.join(path,'dataset-aligned.pkl'), 'wb') as f:
        pickle.dump(dataset_new, f)
    with open(os.path.join(path,'dataset-aligned-para-new.pkl'), 'wb') as f:
        pickle.dump(dataset_new_para, f)


def main():
    dataset = json.load(open('./dataset_new_3.json', 'r', encoding='utf-8'))
    temp = {}
    for one in dataset:
        temp[one['textid']] = one
    type_dict = pickle.load(open('./pmid_type_dict.pkl', 'rb'))
    type2ids = {}
    for (key, value) in type_dict.items():
        if value in type2ids:
            type2ids[value].append(key)
        else:
            type2ids[value] = [key]
    train_valid_test = pickle.load(open('./train_valid_test_diseases.pkl', 'rb'))
    train_diseases = train_valid_test[0]
    valid_diseases = train_valid_test[1]
    test_diseases = train_valid_test[2]
    train_data = []
    for disease in train_diseases:
        train_data += type2ids[disease]
    test_data = []
    for disease in test_diseases:
        test_data += type2ids[disease]
    valid_data = []
    for disease in valid_diseases:
        valid_data += type2ids[disease]
    dataset_train = []
    dataset_test = []
    dataset_valid = []
    for id_one in train_data:
        if id_one not in temp:
            continue
        dataset_train.append(temp[id_one])
    for id_one in test_data:
        if id_one not in temp:
            continue
        dataset_test.append(temp[id_one])
    for id_one in valid_data:
        if id_one not in temp:
            continue
        dataset_valid.append(temp[id_one])
    print('train dataset:')
    aligneddata(dataset_train,'./train')
    print('test dataset:')
    aligneddata(dataset_test,'./test')
    print('valid dataset:')
    aligneddata(dataset_valid,'./valid')


if __name__ == '__main__':
    main()