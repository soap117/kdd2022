from cuda2 import *
import torch
from config import Config
config = Config(20)
from models.units import MyData, get_decoder_att_map, batch_pointer_generation, batch_pointer_decode
from torch.utils.data import DataLoader
from models.retrieval import TitleEncoder, PageRanker, SecEncoder, SectionRanker
from tqdm import tqdm
from transformers import AdamW
import numpy as np
import jieba
from models.units import read_clean_data
from rank_bm25 import BM25Okapi
from models.modeling_gpt2_att import GPT2LMHeadModel
from models.modeling_bart_att import BartForConditionalGeneration
import os

def build(config):
    titles, sections, title2sections, sec2id = read_clean_data(config.data_file)
    corpus = sections
    tokenized_corpus = [jieba.lcut(doc) for doc in corpus]
    bm25_section = BM25Okapi(tokenized_corpus)
    tokenizer = config.tokenizer
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
                                  , collate_fn=train_dataset.collate_fn)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=config.batch_size
                                  , collate_fn=train_dataset.collate_fn)

    title_encoder = TitleEncoder(config)
    modelp = PageRanker(config, title_encoder)
    modelp.cuda()
    models = SectionRanker(config, title_encoder)
    models.cuda()
    modeld = config.modeld.from_pretrained(config.bert_model)
    modeld.cuda()
    optimizer_p = AdamW(modelp.parameters(), lr=config.lr)
    optimizer_s = AdamW(models.parameters(), lr=config.lr)
    optimizer_decoder = AdamW(modeld.parameters(), lr=config.lr)
    loss_func = torch.nn.CrossEntropyLoss(reduction='none')
    return modelp, models, modeld, optimizer_p, optimizer_s, optimizer_decoder, train_dataloader, valid_dataloader, test_dataloader, loss_func, titles, sections, title2sections, sec2id, bm25_title, bm25_section, tokenizer

def train_eval(modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, train_dataloader, valid_dataloader, loss_func):
    min_loss_p = min_loss_s = min_loss_d = 1000
    state = None
    count_s = -1
    count_p = -1
    data_size = len(train_dataloader)
    for epoch in range(config.train_epoch):
        for step, (querys, querys_context, titles, sections, infer_titles, annotations) in zip(
                tqdm(range(data_size)), train_dataloader):
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
            rs_scores = models.infer(query_embedding, infer_section_candidates_pured)
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
            inputs = tokenizer(reference, return_tensors="pt", padding=True)
            ids = inputs['input_ids']
            targets_ = tokenizer(annotations, return_tensors="pt", padding=True)['input_ids']
            adj_matrix = get_decoder_att_map(tokenizer, '[SEP]', ids, scores)
            outputs = model(ids.cuda(), attention_adjust=adj_matrix)
            logits_ = outputs.logits
            len_anno = min(targets_.shape[1], logits_.shape[1])
            logits = logits_[:, 0:len_anno]
            targets = targets_[:, 0:len_anno]
            _, predictions = torch.max(logits, dim=-1)
            results = tokenizer.batch_decode(predictions)
            results = [tokenizer.convert_tokens_to_string(x) for x in results]
            results = [x.replace(' ', '') for x in results]
            results = [x.replace('[PAD]', '') for x in results]
            results = [x.split('[SEP]')[0] for x in results]
            results = [x.replace('[CLS]', '') for x in results]
            logits = logits.reshape(-1, logits.shape[2])
            targets = targets.reshape(-1).to(config.device)
            masks = torch.ones_like(targets)
            masks[torch.where(targets == 0)] = 0
            lossd = (masks*loss_func(logits, targets)).sum()/config.batch_size
            loss = lossd
            if count_s <= 1:
                loss += losss.mean()
            if count_p <= 1:
                loss += lossp.mean()
            optimizer_p.zero_grad()
            optimizer_s.zero_grad()
            optimizer_decoder.zero_grad()
            loss.backward()
            optimizer_p.step()
            optimizer_s.step()
            optimizer_decoder.step()
            if step%100 == 0:
                print('loss P:%f loss S:%f loss D:%f' %(lossp.mean().item(), losss.mean().item(), lossd.item()))
                print(results[0:5])
                print('---------------------------')
        test_loss, eval_ans = test(modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, valid_dataloader, loss_func)
        p_eval_loss = test_loss[0]
        s_eval_loss = test_loss[1]
        d_eval_loss = test_loss[2]
        if d_eval_loss < min_loss_d:
            print(count_p, count_s)
            print('update-all')
            print('New Test Loss D:%f' % (d_eval_loss))
            state = {'epoch': epoch, 'config': config, 'models': models.state_dict(), 'modelp': modelp.state_dict(), 'model': model.state_dict(),
                     'eval_rs': eval_ans}
            torch.save(state, './results/' + config.data_file.replace('.pkl', '_models_full.pkl').replace('data/', ''))
            min_loss_d = d_eval_loss
            for one in eval_ans[0:10]:
                print(one)
            print('+++++++++++++++++++++++++++++++')
        else:
            print(count_p, count_s)
            print('New Test Loss D:%f' % (d_eval_loss))
            for one in eval_ans[0:10]:
                print(one)
            torch.save(state, './results/' + config.data_file.replace('.pkl', '_models_full.pkl').replace('data/', ''))
        if p_eval_loss < min_loss_p:
            print('update-p')
            state['modelp'] = modelp.state_dict()
            min_loss_p = p_eval_loss
            count_p = min(0, epoch-10)
        elif count_p == 2:
            print('p froezen')
            for g in optimizer_p.param_groups:
                g['lr'] = g['lr'] * 0.1
            count_p += 1
        else:
            count_p += 1
        if s_eval_loss < min_loss_s:
            print('update-s')
            state['models'] = models.state_dict()
            min_loss_s = s_eval_loss
            count_s = min(0, epoch-10)
        elif count_s == 2:
            print('s froezen')
            for g in optimizer_s.param_groups:
                g['lr'] = g['lr'] * 0.1
            count_s += 1
        else:
            count_s += 1
    return state

def test(modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, dataloader, loss_func):
    with torch.no_grad():
        modelp.eval()
        models.eval()
        total_loss = []
        eval_ans = []
        data_size = len(dataloader)
        for step, (querys, querys_context, titles, sections, infer_titles, annotations) in zip(
                tqdm(range(data_size)), dataloader):
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
            rs_scores = models.infer(query_embedding, infer_section_candidates_pured)
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
            inputs = tokenizer(reference, return_tensors="pt", padding=True)
            ids = inputs['input_ids']
            targets_ = tokenizer(annotations, return_tensors="pt", padding=True)['input_ids']
            adj_matrix = get_decoder_att_map(tokenizer, '[SEP]', ids, scores)
            outputs = model(ids.cuda(), attention_adjust=adj_matrix)
            logits_ = outputs.logits
            len_anno = min(targets_.shape[1], logits_.shape[1])
            logits = logits_[:, 0:len_anno]
            targets = targets_[:, 0:len_anno]
            _, predictions = torch.max(logits, dim=-1)
            logits = logits.reshape(-1, logits.shape[2])
            targets = targets.reshape(-1).to(config.device)
            masks = torch.ones_like(targets)
            masks[torch.where(targets == 0)] = 0
            results = tokenizer.batch_decode(predictions)
            results = [tokenizer.convert_tokens_to_string(x) for x in results]
            results = [x.replace(' ', '') for x in results]
            results = [x.replace('[PAD]', '') for x in results]
            results = [x.split('[SEP]')[0] for x in results]
            eval_ans += results
            lossd = (masks*loss_func(logits, targets)).sum()/config.batch_size
            loss = [lossp.mean().item(), losss.mean().item(), lossd.item()]
            total_loss.append(loss)
        modelp.train()
        models.train()
        total_loss = np.array(total_loss).mean(axis=0)
        return total_loss, eval_ans


modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, train_dataloader, valid_dataloader, test_dataloader, loss_func, titles, sections, title2sections, sec2id, bm25_title, bm25_section, tokenizer = build(config)
state = train_eval(modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, train_dataloader, valid_dataloader, loss_func)