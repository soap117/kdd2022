from config import config
from models.retrieval import PageRanker, TitleEncoder, SecEncoder, SectionRanker
import pickle
import re
import os
from models.units import neg_sample_title, neg_sample_section
from transformers import GPT2Tokenizer
from models.modeling_gpt2_att import GPT2Model, GPT2LMHeadModel
import torch


keys = {}
keys_section = {}
prob_list = ['研','数','但','为','科', '。', '虚']
for root, dirs, files in os.walk("./data/"):
    if len(dirs) > 0:
        for dir in dirs:
            if dir == "admin":
                continue
            if dir not in ['annotator{}'.format(u) for u in range(50)]:
                continue
            # print(dir)
            for _, __, ___ in os.walk(os.path.join('./data/', dir)):
                for file in ___:
                    if file[0] in '0123456789c':
                        file_path = os.path.join('./data/', dir, file)
                        temp = []
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                            for sec_data in data['contents']:
                                for ann_data in sec_data['tooltips']:
                                    key = ann_data['origin']
                                    if len(key) > 0:
                                        if key in keys:
                                            keys[key] += 1
                                        else:
                                            keys[key] = 1
                                        temp.append(key)
                        keys_section[file_path] = temp
prob_list = ['研','数','但','为','科', '。', '虚']
for key in keys:
    if len(key) <= 1:
        print(key)

sample_data = pickle.load(open('data/mydata.pkl', 'rb'))
keys = []
titles = []
sections = []
title2sections = {}
urls = set()
sec2id = {}
for one in sample_data:
    if len(one['urls']) > 0:
        for tid, (title, url) in enumerate(zip(one['rpsecs'], one['urls'])):
            if len(title) > 0:
                web_title = title[-1]
                web_title = re.sub('_.+', '', web_title)
                web_title = re.sub(' -.+', '', web_title)
                one['rpsecs'][tid][-1] = web_title
                sections += title[0:-1]
                titles.append(web_title)
                if web_title in title2sections and url not in urls:
                    title2sections[web_title] += title[0:-1]
                    urls.add(url)
                elif web_title not in title2sections:
                    title2sections[web_title] = title[0:-1]
                    urls.add(url)

        keys.append(one)
titles = list(set(titles))
sections = list(set(sections))
for k in range(len(sections)-1, -1, -1):
    if len(sections[k]) < 60:
        del sections[k]
for tid, temp in enumerate(sections):
    sec2id[temp] = tid
from rank_bm25 import BM25Okapi
import jieba
import numpy as np
corpus = sections
tokenized_corpus = [jieba.lcut(doc) for doc in corpus]
bm25_section = BM25Okapi(tokenized_corpus)

corpus = titles
tokenized_corpus = [jieba.lcut(doc) for doc in corpus]
bm25_title = BM25Okapi(tokenized_corpus)

sample_query = []
sample_title_candidates = []
sample_section_candidates = []
sample_candidates = []
infer_title_candidates = []
for key in keys[0:4]:
    if len(key['rpsecs'][0]) <= 0:
        continue
    print('****************************************************')
    print(key['key'])
    sample_query.append(key['key'])
    key_cut = jieba.lcut(key['key'])
    infer_titles = bm25_title.get_top_n(key_cut, titles, 4)
    infer_title_candidates.append(infer_titles)
    neg_titles = neg_sample_title(key['key'], [x[-1] for x in key['rpsecs']], titles, 5)
    neg_sections = neg_sample_section(key['key'], key['rsecs'], sections, 5, bm25_section)
    pos_section = key['rsecs'][np.random.randint(len(key['rsecs']))]
    pos_title = key['rpsecs'][np.random.randint(len(key['rpsecs']))][-1]
    sample_title_candidates.append([pos_title] + neg_titles)
    sample_section_candidates.append([pos_section] + neg_sections)
    print('____________________________________________________')
    print(neg_titles)
    print('____________________________________________________')
    for x in neg_sections:
        print(x)
        print('____________________________________________________')



title_encoder = TitleEncoder(config)
modelp = PageRanker(config, title_encoder)
modelp.cuda()
rs = modelp(sample_query, sample_title_candidates)
rs2 = modelp.infer(sample_query, infer_title_candidates)
rs2 = torch.topk(rs2, 3, dim=1)
scores_title = rs2[0]
inds = rs2[1].cpu().numpy()
infer_title_candidates_pured = []
infer_section_candidates_pured = []
mapping_title = np.zeros([len(sample_query), 3, 4])
for query, bid in zip(sample_query, range(len(inds))):
    temp = []
    temp2 = []
    temp3 = []
    for nid, cid in enumerate(inds[bid]):
        temp.append(infer_title_candidates[bid][cid])
        temp2 += title2sections[infer_title_candidates[bid][cid]]
        temp3 += [nid for x in title2sections[infer_title_candidates[bid][cid]]]
    temp2_id = []
    for t_sec in temp2:
        if t_sec in sec2id:
            temp2_id.append(sec2id[t_sec])
    key_cut = jieba.lcut(query)
    ls_scores = bm25_section.get_batch_scores(key_cut, temp2_id)
    cindex = np.argsort(ls_scores)[::-1][0:4]
    temp2_pured = []
    temp3_pured = []
    for oid, one in enumerate(cindex):
        temp2_pured.append(temp2[one])
        temp3_pured.append(temp3[one])
        mapping_title[bid, temp3[one], oid] = 1
    while len(temp2_pured) < 4:
        temp2_pured.append('')
        temp3_pured.append(-1)

    infer_title_candidates_pured.append(temp)
    infer_section_candidates_pured.append(temp2_pured)

mapping = torch.FloatTensor(mapping_title).to(config.device)
scores_title = scores_title.unsqueeze(1)
scores_title = scores_title.matmul(mapping).squeeze(1)
sec_encoder = SecEncoder(config)
models = SectionRanker(config, sec_encoder)
models.cuda()
rs = models(sample_query, sample_section_candidates)
rs_scores = models.infer(sample_query, infer_section_candidates_pured)
scores = scores_title * rs_scores
rs2 = torch.topk(scores, 2, dim=1)
scores = rs2[0]
reference = []
inds_sec = rs2[1].cpu().numpy()
for bid in range(len(inds_sec)):
    temp = []
    for indc in inds_sec[bid]:
        temp.append(infer_section_candidates_pured[bid][indc])
    temp = ' [SEP] '.join(temp)
    reference.append(temp[0:100])

#[B, L] [B, L]
from models.bert_tokenizer import BertTokenizer
#from transformers.models.gpt2 import GPT2LMHeadModel
tokenizer = BertTokenizer(vocab_file='./GPT2Chinese/vocab.txt', max_len=10)
model = GPT2LMHeadModel.from_pretrained("./GPT2Chinese/")
model.train()
model.cuda()
#reference = ['你是傻逼 [SEP] 是吧', '你是傻逼 SEP 是吧']
inputs = tokenizer(reference, return_tensors="pt", padding=True)
ids = inputs['input_ids']
tokenizer._batch_encode_plus(reference)
def getting_decoder_att_map(tokenizer, sep, ids, scores):
    spe_seq = tokenizer.encode(sep)
    mapping = np.zeros([len(ids), scores.shape[1], ids.shape[1]])
    for bindex, (bscore, bids) in enumerate(zip(scores, ids)):
        i = 0
        c_father = 0
        while i < len(bids):
            mapping[bindex, c_father, i] = 1
            i += 1
            k = 0
            while k<len(spe_seq) and i<len(bids) and (spe_seq[k]==bids[i]):
                i += 1
                k += 1
            if k == len(spe_seq):
                c_father += 1
    mapping = torch.FloatTensor(mapping).to(config.device)
    scores = scores.unsqueeze(1)
    scores = scores.matmul(mapping).squeeze(1)
    return scores
adj_matrix = getting_decoder_att_map(tokenizer, 'SEP', ids, scores)
stemp_adj = torch.zeros_like(ids) + 0.5
#attention_adjust=adj_matrix
outputs = model(ids.cuda(), attention_adjust=adj_matrix)
fake_loss = torch.mean(outputs.logits[:,:,2])
optimizer = torch.optim.Adam(modelp.parameters())
optimizer.zero_grad()
fake_loss.backward()
optimizer.step()
last_hidden_states = outputs.last_hidden_state