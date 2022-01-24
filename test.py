import torch
import pickle
sample_data = pickle.load(open('./data/valid.pkl', 'rb'))
keys = []
for one in sample_data:
    if len(one['urls']) > 0:
        if len(one['rpsecs'][0]) <= 0 or len(one['key']) < 1:
            continue
        keys.append(one)
save_data = torch.load('./results/best_model.bin', map_location=torch.device('cuda:0'))
test_results = save_data['eval_rs']
for g, p in zip(keys, test_results):
    g_ans = g['anno']
    if len(g_ans) == 0:
        print('here')
    p_ans = p
    print("%s||%s" %(g_ans, p_ans))