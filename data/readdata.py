import json
import pickle
import os
import random
import re

def read_data_ori():
    data_path = "/"
    src = []
    tar = []
    done = set()
    for root, dirs, files in os.walk(data_path):
        if len(dirs) > 0:
            for dir in dirs:
                print(dir)
                if dir=="admin":
                    continue
                for _, __, ___ in os.walk(os.path.join(data_path,dir)):
                    print(len(___))
                    for file in ___:
                        if file[0] in '0123456789c':
                            with open(os.path.join(data_path,dir,file),'rb') as f:
                                data = pickle.load(f)
                                if data['textid'] not in done:
                                    done.add(data['textid'])
                                    src.append(data['src'])
                                    tar.append(data['tar'])
    return src, tar

def read_data():
    raw_data = json.load(open('dataset_new_2.json', 'r', encoding='utf-8'))
    src = []
    tar = []
    for data in raw_data:
        modified_src = ''
        for content in data['contents']:
            sen_text = content['text']
            modified_src += sen_text
        src.append(modified_src)
        tar.append(data['tar'])
    return src, tar


def main():
    src_all, tar_all = read_data()
    total = len(src_all)
    print(total)
    dataset = [(u,v) for u,v in zip(src_all,tar_all)]
    random.seed(2021)
    random.shuffle(dataset)
    with open('./dataset-all.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    with open('./train/dataset.pkl', 'wb') as f:
        pickle.dump(dataset[:int(total/10*8)], f)
    with open('./test/dataset.pkl', 'wb') as f:
        pickle.dump(dataset[int(total/10*8):int(total/10*9)], f)
    with open('./valid/dataset.pkl', 'wb') as f:
        pickle.dump(dataset[int(total/10*9):], f)
    print('train:test:valid = {}:{}:{}'.format(int(total/10*8),int(total/10*9)-int(total/10*8),total-int(total/10*9)))

if __name__ == '__main__':
    main()