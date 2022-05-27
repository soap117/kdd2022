import json
import pickle

dataset = json.load(open('../data/dataset_new_3.json', 'r', encoding='utf-8'))
total = len(dataset)
dataset_train = dataset[:int(total/10*8)]
dataset_test_dev = dataset[int(total/10*8):]

train_keys = set()
for data in dataset_train:
    for content in data['contents']:
        for tooltip in content['tooltips']:
            train_keys.add(tooltip['origin'])

print(len(train_keys))

with open('./data/train_keys.pkl','wb') as f:
    pickle.dump(train_keys, f)