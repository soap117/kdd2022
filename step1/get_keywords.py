import json
import pickle

dataset = json.load(open('../data/dataset_new_3.json', 'r', encoding='utf-8'))
temp = {}
for one in dataset:
    temp[one['textid']] = one
type_dict = pickle.load(open('../data/pmid_type_dict.pkl', 'rb'))
type2ids = {}
for (key, value) in type_dict.items():
    if value in type2ids:
        type2ids[value].append(key)
    else:
        type2ids[value] = [key]
diseases_list = list(type2ids.keys())
total = len(diseases_list)
import random
random.shuffle(diseases_list)
train_diseases = diseases_list[:int(total / 10 * 7)]
valid_diseases = diseases_list[int(total / 10 * 7):int(total / 10 * 8)]
test_diseases = diseases_list[int(total / 10 * 8):]
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

train_keys = set()
for data in dataset_train:
    for content in data['contents']:
        for tooltip in content['tooltips']:
            train_keys.add(tooltip['origin'])

print(len(train_keys))
test_keys = set()
for data in dataset_test:
    for content in data['contents']:
        for tooltip in content['tooltips']:
            test_keys.add(tooltip['origin'])

print(len(test_keys))
unique_test_keys = test_keys.difference(train_keys)
print(len(unique_test_keys))
#print(unique_test_keys)

with open('./data/train_keys.pkl','wb') as f:
    pickle.dump(train_keys, f)
with open('./data/test_keys.pkl','wb') as f:
    pickle.dump(test_keys, f)
with open('./data/unique_test_keys.pkl','wb') as f:
    pickle.dump(unique_test_keys, f)