import os
import pickle

import torch

from step1.modeling_cbert import BertForTokenClassification
from transformers import BertTokenizer
from  step1.utils.dataset import obtain_indicator
import numpy as np
batch_size = 4
import jieba
from models.units import read_clean_data, get_decoder_att_map
from config import config
from rank_bm25 import BM25Okapi
from models.retrieval import TitleEncoder, PageRanker, SectionRanker
titles, sections, title2sections, sec2id = read_clean_data(config.data_file)
corpus = sections
tokenized_corpus = [jieba.lcut(doc) for doc in corpus]
bm25_section = BM25Okapi(tokenized_corpus)
step2_tokenizer = config.tokenizer
step2_tokenizer.model_max_length = 300
corpus = titles
tokenized_corpus = [jieba.lcut(doc) for doc in corpus]
bm25_title = BM25Okapi(tokenized_corpus)
save_data = torch.load('./results/' + config.data_file.replace('.pkl', '_models_full.pkl').replace('data/', ''))
save_step1_data = torch.load('./results/' + 'best_save.data')

bert_model = 'hfl/chinese-bert-wwm-ext'
model_step1 = BertForTokenClassification.from_pretrained(bert_model, num_labels=5)
model_step1.load_state_dict(save_step1_data['para'])
model_step1.eval()
step1_tokenizer = BertTokenizer.from_pretrained(bert_model)
step1_tokenizer.model_max_length = 512

title_encoder = TitleEncoder(config)
modelp = PageRanker(config, title_encoder)
modelp.load_state_dict(save_data['modelp'])
modelp.cuda()
modelp.eval()
models = SectionRanker(config, title_encoder)
models.load_state_dict(save_data['models'])
models.cuda()
models.eval()
modeld = config.modeld.from_pretrained(config.bert_model)
modeld.load_state_dict(save_data['model'])
modeld.cuda()
modeld.eval()
def obtain_step2_input(pre_labels, src, src_ids, step1_tokenizer):
    input_list = [[],[],[], []]
    l = 0
    r = -1
    while src_ids[r] != 511:
        r += 1
    for c_id in range(len(src_ids)):
        if src_ids[c_id] == 511:
            l = r + 1
            r = l+1
            while r<len(src_ids) and src_ids[r] != 511:
                r += 1
        if pre_labels[c_id] == 1:
            l_k = c_id
            r_k = l_k+1
            while r_k<len(pre_labels) and pre_labels[r_k] != 0 and pre_labels[r_k] != 4:
                r_k += 1
            if pre_labels[r_k] == 4:
                r_k += 1
            templete = src_ids[l_k:r_k]
            tokens = step1_tokenizer.convert_ids_to_tokens(templete)
            key = step1_tokenizer.convert_tokens_to_string(tokens).replace(' ', '')
            context = step1_tokenizer.convert_ids_to_tokens(src_ids[l:r+1])
            context = step1_tokenizer.convert_tokens_to_string(context).replace('[CLS]', '').replace('[SEP]', '')
            key_cut = jieba.lcut(key)
            infer_titles = bm25_title.get_top_n(key_cut, titles, config.infer_title_range)
            input_list[0].append(key)
            input_list[1].append(context)
            input_list[2].append(infer_titles)
            input_list[3].append((l_k, r_k))
    return input_list





def pipieline(path_from):
    with open(os.path.join(path_from,'dataset-aligned.pkl'), 'rb') as f:
        dataset_aligned = pickle.load(f)
    srcs = []
    tars = []
    for dp in dataset_aligned:
        srcs.append(dp[0])
        tars.append(dp[1])
    for src, tar in zip(srcs[0:50], tars[0:50]):
        src_ = step1_tokenizer([src], return_tensors="pt", padding=True, truncation=True)
        x_ids = src_['input_ids']
        x_mask = src_['attention_mask']
        x_indicator = torch.zeros_like(x_ids)
        outputs = model_step1(x_ids, attention_mask=x_mask, existing_indicates=x_indicator)
        logits = outputs.logits
        pre_label_0 = np.argmax(logits.detach().cpu().numpy(), axis=2)
        x_indicator = obtain_indicator(x_ids[0], pre_label_0[0])
        x_indicator = torch.LongTensor(x_indicator).unsqueeze(0)
        outputs = model_step1(x_ids, attention_mask=x_mask, existing_indicates=x_indicator)
        logits = outputs.logits
        pre_label_f = np.argmax(logits.detach().cpu().numpy(), axis=2)
        step2_input = obtain_step2_input(pre_label_f[0], src, x_ids[0], step1_tokenizer)
        querys = step2_input[0]
        contexts = step2_input[1]
        infer_titles = step2_input[2]
        key_pos = step2_input[3]
        dis_scores, query_embeddings = modelp.infer_pipe(step2_input[0], step2_input[1], step2_input[2])
        rs_title = torch.topk(dis_scores, config.infer_title_select, dim=1)
        scores_title = rs_title[0]
        inds = rs_title[1].cpu().numpy()
        infer_title_candidates_pured = []
        infer_section_candidates_pured = []
        mapping_title = np.zeros([len(querys), config.infer_title_select, config.infer_section_range])
        for query, bid in zip(querys, range(len(inds))):
            temp = []
            temp2 = []
            temp3 = []
            count = 0
            for nid, cid in enumerate(inds[bid]):
                temp.append(infer_titles[bid][cid])
                temp2 += title2sections[infer_titles[bid][cid]]
                temp3 += [nid for x in title2sections[infer_titles[bid][cid]]]
            temp2_id = []
            for t_sec in temp2:
                if t_sec in sec2id:
                    temp2_id.append(sec2id[t_sec])
            key_cut = jieba.lcut(query)
            ls_scores = bm25_section.get_batch_scores(key_cut, temp2_id)
            cindex = np.argsort(ls_scores)[::-1][0:config.infer_section_range]
            temp2_pured = []
            for oid, one in enumerate(cindex):
                temp2_pured.append(temp2[one])
                mapping_title[bid, temp3[one], oid] = 1
            while len(temp2_pured) < config.infer_section_range:
                temp2_pured.append('')

            infer_title_candidates_pured.append(temp)
            infer_section_candidates_pured.append(temp2_pured)

        mapping = torch.FloatTensor(mapping_title).to(config.device)
        scores_title = scores_title.unsqueeze(1)
        scores_title = scores_title.matmul(mapping).squeeze(1)
        rs_scores = models.infer(query_embeddings, infer_section_candidates_pured)
        scores = scores_title * rs_scores
        rs2 = torch.topk(scores, config.infer_section_select, dim=1)
        scores = rs2[0]
        reference = []
        inds_sec = rs2[1].cpu().numpy()
        for bid in range(len(inds_sec)):
            temp = [querys[bid]]
            for indc in inds_sec[bid]:
                temp.append(infer_section_candidates_pured[bid][indc][0:config.maxium_sec])
            temp = ' [SEP] '.join(temp)
            reference.append(temp[0:500])
        inputs = step2_tokenizer(reference, return_tensors="pt", padding=True, truncation=True)
        ids = inputs['input_ids']
        adj_matrix = get_decoder_att_map(step2_tokenizer, '[SEP]', ids, scores)
        outputs = modeld(ids.cuda(), attention_adjust=adj_matrix)
        logits_ = outputs.logits
        logits = logits_
        _, predictions = torch.max(logits, dim=-1)
        results = step2_tokenizer.batch_decode(predictions)
        results = [step2_tokenizer.convert_tokens_to_string(x) for x in results]
        results = [x.replace(' ', '') for x in results]
        results = [x.replace('[PAD]', '') for x in results]
        results = [x.replace('[UNK]', '') for x in results]
        results = [x.split('[SEP]')[0] for x in results]
        results = [x.replace('[CLS]', '') for x in results]
        new_src = ''
        l = 0
        for result, pos in zip(results, key_pos):
            context = step1_tokenizer.convert_ids_to_tokens(x_ids[0][l:pos[0]])
            context = step1_tokenizer.convert_tokens_to_string(context).replace('[CLS]', '').replace('[SEP]', '').replace(' ', '')
            new_src += context
            new_src += '<{}>'.format(result)
            l = pos[1]
        context = step1_tokenizer.convert_ids_to_tokens(x_ids[0][l:])
        context = step1_tokenizer.convert_tokens_to_string(context).replace('[CLS]', '').replace('[SEP]', '').replace(
            ' ', '')
        new_src += context
        print(new_src)





pipieline('./data/test')
