from cuda import *
import torch
from config import config
from models.units import MyData
from models.bert_tokenizer import BertTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW
import numpy as np
import jieba
from models.units import read_clean_data
from rank_bm25 import BM25Okapi
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
smooth = SmoothingFunction()
import os
def build(config):
    save_data = torch.load('./results/' + config.data_file.replace('.pkl', '_models.pkl').replace('data/', ''))
    tokenizer = BertTokenizer(vocab_file='./GPT2Chinese/vocab.txt', do_lower_case=False, never_split=['[SEP]'])
    titles, sections, title2sections, sec2id = read_clean_data(config.data_file)
    corpus = sections
    tokenized_corpus = [jieba.lcut(doc) for doc in corpus]
    bm25_section = BM25Okapi(tokenized_corpus)

    corpus = titles
    tokenized_corpus = [jieba.lcut(doc) for doc in corpus]
    bm25_title = BM25Okapi(tokenized_corpus)
    train_dataset = None
    if os.path.exists(config.data_file.replace('.pkl', '_train_dataset.pkl')):
        valid_dataset = torch.load(config.data_file.replace('.pkl', '_valid_dataset.pkl'))
        test_dataset = torch.load(config.data_file.replace('.pkl', '_test_dataset.pkl'))
    else:
        valid_dataset = MyData(config, tokenizer, config.data_file.replace('.pkl', '_valid_dataset_raw.pkl'), titles,
                               sections, title2sections, sec2id,
                               bm25_title,
                               bm25_section)
        test_dataset = MyData(config, tokenizer, config.data_file.replace('.pkl', '_test_dataset_raw.pkl'), titles,
                              sections, title2sections, sec2id, bm25_title,
                              bm25_section)
        torch.save(train_dataset, config.data_file.replace('.pkl', '_train_dataset.pkl'))
        torch.save(valid_dataset, config.data_file.replace('.pkl', '_valid_dataset.pkl'))
        torch.save(test_dataset, config.data_file.replace('.pkl', '_test_dataset.pkl'))
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=config.batch_size
                                  , collate_fn=valid_dataset.collate_fn_test)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=config.batch_size
                                  , collate_fn=test_dataset.collate_fn_test)

    modelp = save_data['modelp']
    models = save_data['models']
    model = save_data['model']
    optimizer_p = AdamW(modelp.parameters(), lr=config.lr)
    optimizer_s = AdamW(models.parameters(), lr=config.lr)
    optimizer_decoder = None
    loss_func = torch.nn.CrossEntropyLoss()
    return modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, None, valid_dataloader, test_dataloader, loss_func, titles, sections, title2sections, sec2id, bm25_title, bm25_section, tokenizer

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
        for step, (querys, querys_context, titles, sections, infer_titles, annotations_ids, pos_titles, pos_sections) in tqdm(enumerate(dataloader)):
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
                if count == 0:
                    print('Failed Examples:')
                    print(query)
                    print(titles[bid][0])
                    print(temp)
                    print('------------------------------------')
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
            scores = scores_title + rs_scores
            rs2 = torch.topk(scores, config.infer_section_select, dim=1)
            scores = rs2[0]
            reference = []
            inds_sec = rs2[1].cpu().numpy()
            for query, bid in zip(querys, range(len(inds))):
                total_s += 1
                temp = []
                for indc in inds_sec[bid]:
                    temp.append(infer_section_candidates_pured[bid][indc][0:config.maxium_sec])
                if check(query, temp, pos_sections[bid], secs=True):
                    tp_s += 1
                else:
                    print('Failed Examples:')
                    print(querys[bid])
                    print(pos_sections[bid])
                    print(temp)
                    print('++++++++++++++++++++++++++++++++++++')
                temp = ' [SEP] '.join(temp)
                reference.append(temp[0:1000])
            loss = (lossp.mean() + losss.mean())
            total_loss.append(loss.item())
        modelp.train()
        models.train()
        print('accuracy title: %f accuracy section: %f' %(tp/total, tp_s/total_s))
        return np.array(total_loss).mean(), eval_ans


modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, train_dataloader, valid_dataloader, test_dataloader, loss_func, titles, sections, title2sections, sec2id, bm25_title, bm25_section, tokenizer = build(config)
test(modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, valid_dataloader, loss_func)