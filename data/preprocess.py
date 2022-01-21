import os
import pickle
if os.path.exists('./mydata_done_baidu.pkl'):
    with open('./mydata_done_baidu.pkl','rb') as f:
        mark_done = pickle.load(f)
    with open('./mydata_new_baidu.pkl', 'rb') as f:
        my_data = pickle.load(f)
    with open('./mydata_url2secs_new_baidu.pkl', 'rb') as f:
        url2secs = pickle.load(f)
    with open('./mydata_url_new_baidu.pkl', 'rb') as f:
        url_done = pickle.load(f)
    with open('./paths.pkl', 'rb') as f:
        paths = pickle.load(f)
path2file = {}
for dp in my_data:
    for eid, rpsec in enumerate(dp['rpsecs']):
        if len(rpsec) == 0:
            dp['rpsecs'][eid] = url2secs[dp['urls'][eid]]
    path2file[dp['file'][2:]] = dp
with open('./mydata_new_baidu.pkl', 'wb') as f:
    pickle.dump(my_data, f)
names = ['train', 'valid', 'test']
sets = {}
for name in names:
    sets[name] = []
    with open(name+'/dataset.pkl', 'rb') as f:
        data_list = pickle.load(f)
        for dp_c in data_list:
            sets[name].append(dp_c[3])
for name in names:
    plist = sets[name]
    temp = []
    for path in plist:
        if path[7:] in path2file:
            temp.append(path2file[path[7:]])
    with open('./'+name+'.pkl', 'wb') as f:
        pickle.dump(temp, f)

