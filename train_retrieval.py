from cuda import *
import torch
from config import Config
config = Config(64)
from models.units import MyData
from torch.utils.data import DataLoader
from models.retrieval import TitleEncoder, PageRanker, SecEncoder, SectionRanker
from tqdm import tqdm
from transformers import AdamW
import numpy as np
import jieba
from models.units import read_clean_data
from rank_bm25 import BM25Okapi
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
smooth = SmoothingFunction()
def build(config):
    tokenizer = config.tokenizer
    titles, sections, title2sections, sec2id = read_clean_data(config.data_file)
    corpus = sections
    tokenized_corpus = [jieba.lcut(doc) for doc in corpus]
    bm25_section = BM25Okapi(tokenized_corpus)

    corpus = titles
    tokenized_corpus = [jieba.lcut(doc) for doc in corpus]
    bm25_title = BM25Okapi(tokenized_corpus)
    if os.path.exists(config.data_file.replace('.pkl', '_train_dataset.pkl')):
        train_dataset = torch.load(config.data_file.replace('.pkl', '_train_dataset.pkl'))
        valid_dataset = torch.load(config.data_file.replace('.pkl', '_valid_dataset.pkl'))
        test_dataset = torch.load(config.data_file.replace('.pkl', '_test_dataset.pkl'))
    else:
        train_dataset = MyData(config, tokenizer, config.data_file.replace('.pkl', '_train_dataset_raw.pkl'), titles, sections, title2sections, sec2id, bm25_title, bm25_section)
        valid_dataset = MyData(config, tokenizer, config.data_file.replace('.pkl', '_valid_dataset_raw.pkl'), titles, sections, title2sections, sec2id,
                               bm25_title,
                               bm25_section)
        test_dataset = MyData(config, tokenizer, config.data_file.replace('.pkl', '_test_dataset_raw.pkl'), titles, sections, title2sections, sec2id, bm25_title,
                              bm25_section)
        torch.save(train_dataset, config.data_file.replace('.pkl', '_train_dataset.pkl'))
        torch.save(valid_dataset, config.data_file.replace('.pkl', '_valid_dataset.pkl'))
        torch.save(test_dataset, config.data_file.replace('.pkl', '_test_dataset.pkl'))

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size
                                  , collate_fn=train_dataset.collate_fn)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=config.batch_size
                                  , collate_fn=train_dataset.collate_fn_test)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=config.batch_size
                                  , collate_fn=train_dataset.collate_fn_test)

    title_encoder = TitleEncoder(config)
    modelp = PageRanker(config, title_encoder)
    modelp.cuda()
    models = SectionRanker(config, title_encoder)
    models.cuda()
    model = None
    optimizer_p = AdamW(modelp.parameters(), lr=config.lr)
    optimizer_s = AdamW(models.parameters(), lr=config.lr)
    optimizer_decoder = None
    loss_func = torch.nn.CrossEntropyLoss()
    return modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, train_dataloader, valid_dataloader, test_dataloader, loss_func, titles, sections, title2sections, sec2id, bm25_title, bm25_section, tokenizer

def train_eval(modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, train_dataloader, valid_dataloader, loss_func):
    min_loss_p = min_loss_s = min_loss_d = 1000
    state = {}
    data_size = len(train_dataloader)
    for epoch in range(config.train_epoch):
        for step, (querys, querys_context, titles, sections, infer_titles, annotations) in zip(tqdm(range(data_size)), train_dataloader):
            dis_final, lossp, query_embedding = modelp(querys, querys_context, titles)
            dis_final, losss = models(query_embedding, sections)
            rs2 = modelp.infer(query_embedding, infer_titles)
            rs2 = torch.topk(rs2, config.infer_title_select, dim=1)
            scores_title = rs2[0]
            inds = rs2[1].cpu().numpy()
            infer_title_candidates_pured = []
            infer_section_candidates_pured = []
            mapping_title = np.zeros([len(querys), config.infer_title_select, config.infer_section_range])
            for query, bid in zip(querys, range(len(inds))):
                temp = []
                temp2 = []
                temp3 = []
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
            scores_sentence = models.infer(query_embedding, infer_section_candidates_pured)
            scores = scores_title * scores_sentence
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

            loss = lossp.mean() + losss.mean()
            optimizer_p.zero_grad()
            optimizer_s.zero_grad()
            loss.backward()
            optimizer_p.step()
            optimizer_s.step()
            if step%500 == 0:
                print('loss P:%f loss S:%f' %(lossp.mean().item(), losss.mean().item()))
        test_loss, eval_ans = test(modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, valid_dataloader, loss_func)
        p_eval_loss = test_loss[0]
        s_eval_loss = test_loss[1]
        if p_eval_loss + s_eval_loss <= min_loss_p + min_loss_s:
            print('New Test Loss:%f' % (p_eval_loss+s_eval_loss))
        if p_eval_loss < min_loss_p:
            print('update-p')
            state['modelp'] = modelp.state_dict()
            min_loss_p = p_eval_loss
            count_p = 0
            torch.save(state, './results/' + config.data_file.replace('.pkl', '_models.pkl').replace('data/', ''))
        elif count_p > 1:
            print('p froezen')
            for g in optimizer_p.param_groups:
                g['lr'] = g['lr'] * 0.1
        else:
            count_p += 1
        if s_eval_loss < min_loss_s:
            print('update-s')
            state['models'] = models.state_dict()
            min_loss_s = s_eval_loss
            count_s = 0
            torch.save(state, './results/' + config.data_file.replace('.pkl', '_models.pkl').replace('data/', ''))
        elif count_s > 1:
            print('s froezen')
            for g in optimizer_s.param_groups:
                g['lr'] = g['lr'] * 0.1
        else:
            count_s += 1
        if count_s > 2 and count_p > 2:
            print('early stop')
            return state
    return state

def check(query, infer_titles, pos_titles, secs=False):
    for pos_title in pos_titles:
        for infer_title in infer_titles:
            key_cut = list(pos_title)
            candidata_title = list(infer_title)
            if secs:
                if min(len(key_cut), len(candidata_title)) == 1:
                    can_simi = sentence_bleu([candidata_title], key_cut, weights=(1.0, 0.0),
                                             smoothing_function=smooth.method1)
                elif min(len(key_cut), len(candidata_title)) == 2:
                    can_simi = sentence_bleu([candidata_title], key_cut, weights=(0.5, 0.5),
                                             smoothing_function=smooth.method1)
                else:
                    can_simi = sentence_bleu([candidata_title], key_cut, weights=(0.3333, 0.3333, 0.3333),
                                             smoothing_function=smooth.method1)
                if can_simi > 0.5 or pos_title in infer_title:
                    return True
            else:
                if min(len(key_cut), len(candidata_title)) == 1:
                    can_simi = sentence_bleu([candidata_title], key_cut, weights=(1.0, 0.0),
                                             smoothing_function=smooth.method1)
                else:
                    can_simi = sentence_bleu([candidata_title], key_cut, weights=(0.5, 0.5),
                                             smoothing_function=smooth.method1)
                if can_simi > 0.5 or pos_title in infer_title or query in infer_title:
                    return True
    return False

def test(modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, dataloader, loss_func):
    with torch.no_grad():
        modelp.eval()
        models.eval()
        total_loss = []
        eval_ans = []
        tp = 0
        total = 0
        tp_s = 0
        total_s = 0
        for step, (querys, querys_context, titles, sections, infer_titles, annotations, pos_titles, pos_sections) in tqdm(enumerate(dataloader)):
            dis_final, lossp, query_embedding = modelp(querys, querys_context, titles)
            dis_final, losss = models(query_embedding, sections)
            rs2 = modelp.infer(query_embedding, infer_titles)
            rs2 = torch.topk(rs2, config.infer_title_select, dim=1)
            scores_title = rs2[0]
            inds = rs2[1].cpu().numpy()
            infer_title_candidates_pured = []
            infer_section_candidates_pured = []
            mapping_title = np.zeros([len(querys), config.infer_title_select, config.infer_section_range])
            for query, bid in zip(querys, range(len(inds))):
                total += 1
                temp = []
                temp2 = []
                temp3 = []
                count = 0
                for nid, cid in enumerate(inds[bid]):
                    temp.append(infer_titles[bid][cid])
                    temp2 += title2sections[infer_titles[bid][cid]]
                    temp3 += [nid for x in title2sections[infer_titles[bid][cid]]]
                if check(query, temp, pos_titles[bid]):
                    count += 1
                temp2_id = []
                tp += count
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
            rs_scores = models.infer(query_embedding, infer_section_candidates_pured)
            scores = scores_title * rs_scores
            rs2 = torch.topk(scores, config.infer_section_select, dim=1)
            scores = rs2[0]
            reference = []
            inds_sec = rs2[1].cpu().numpy()
            for query, bid in zip(querys, range(len(inds))):
                total_s += 1
                temp = [querys[bid]]
                for indc in inds_sec[bid]:
                    temp.append(infer_section_candidates_pured[bid][indc][0:config.maxium_sec])
                if check(query, temp, pos_sections[bid], secs=True):
                    tp_s += 1
                temp = ' [SEP] '.join(temp)
                reference.append(temp[0:1000])
            loss = (lossp.mean() + losss.mean())
            total_loss.append(loss.item())
        modelp.train()
        models.train()
        print('accuracy title: %f accuracy section: %f' %(tp/total, tp_s/total_s))
        return (-tp/total, -tp_s/total_s), eval_ans


modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, train_dataloader, valid_dataloader, test_dataloader, loss_func, titles, sections, title2sections, sec2id, bm25_title, bm25_section, tokenizer = build(config)
state = train_eval(modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, train_dataloader, valid_dataloader, loss_func)