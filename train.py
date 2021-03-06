import torch
from config import config
from models.units import MyData, get_decoder_att_map, batch_pointer_generation, batch_pointer_decode
from torch.utils.data import DataLoader
from models.retrieval import TitleEncoder, PageRanker, SecEncoder, SectionRanker
from models.modeling_p2p_att import PointerNet
from tqdm import tqdm
from transformers import AdamW
import numpy as np
import jieba
from models.units import read_clean_data
from rank_bm25 import BM25Okapi


def build(config):
    tokenizer = config.title_tokenizer
    titles, sections, title2sections, sec2id = read_clean_data(config.data_file)
    corpus = sections
    tokenized_corpus = [jieba.lcut(doc) for doc in corpus]
    bm25_section = BM25Okapi(tokenized_corpus)

    corpus = titles
    tokenized_corpus = [jieba.lcut(doc) for doc in corpus]
    bm25_title = BM25Okapi(tokenized_corpus)
    train_dataset = MyData(config, tokenizer, 'data/train.pkl', titles, sections, title2sections, sec2id, bm25_title, bm25_section)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size
                                  , collate_fn=train_dataset.collate_fn)
    valid_dataset = MyData(config, tokenizer, 'data/valid.pkl', titles, sections, title2sections, sec2id, bm25_title,
                           bm25_section)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=config.batch_size
                                  , collate_fn=train_dataset.collate_fn)
    test_dataset = MyData(config, tokenizer, 'data/test.pkl', titles, sections, title2sections, sec2id, bm25_title,
                           bm25_section)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=config.batch_size
                                  , collate_fn=train_dataset.collate_fn)

    title_encoder = TitleEncoder(config)
    modelp = PageRanker(config, title_encoder)
    modelp.cuda()
    section_encoder = SecEncoder(config)
    models = SectionRanker(config, section_encoder)
    models.cuda()
    model = PointerNet(128)
    model.train()
    model.cuda()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not (any(nd in n for nd in no_decay) and 'encoder' in n)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'encoder' in n], 'weight_decay': 0.0}
    ]
    optimizer_p = AdamW(modelp.parameters(), lr=config.lr)
    optimizer_s = AdamW(models.parameters(), lr=config.lr)
    optimizer_decoder = AdamW(optimizer_grouped_parameters, lr=config.lr*0.1)
    loss_func = torch.nn.CrossEntropyLoss()
    return modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, train_dataloader, valid_dataloader, test_dataloader, loss_func, titles, sections, title2sections, sec2id, bm25_title, bm25_section, tokenizer

def train_eval(modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, train_dataloader, valid_dataloader, loss_func):
    min_loss_p = min_loss_s = min_loss_d = 1000
    for epoch in range(config.train_epoch):
        for step, (querys, titles, sections, infer_titles, annotations_ids) in tqdm(enumerate(train_dataloader)):
            dis_final, lossp = modelp(querys, titles)
            dis_final, losss = models(querys, sections)
            rs2 = modelp.infer(querys, infer_titles)
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
            rs_scores = models.infer(querys, infer_section_candidates_pured)
            scores = scores_title + rs_scores
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
            adj_matrix = get_decoder_att_map(tokenizer, '[SEP]', ids, scores)
            outputs, outputs_logits, pointers = model(ids.cuda(), attention_adjust=adj_matrix, out_length=2*annotations_ids['input_ids'].shape[1])
            logits_ = outputs_logits
            targets_ = annotations_ids['input_ids']
            targets_ = batch_pointer_generation(ids, targets_)
            len_anno = min(targets_.shape[1], logits_.shape[1])
            logits = logits_[:, 0:len_anno]
            targets = targets_[:, 0:len_anno]
            pointers = pointers[:, 0:len_anno]
            predictions = batch_pointer_decode(ids, pointers)
            logits = logits.reshape(-1, logits.shape[2])
            targets = targets.reshape(-1).to(config.device)
            try:
                lossd = loss_func(logits, targets)
            except Exception as e:
                print(logits_.shape)
                print(targets_.shape)
                print(e)
            loss = lossp.mean() + losss.mean() + lossd
            optimizer_p.zero_grad()
            optimizer_s.zero_grad()
            optimizer_decoder.zero_grad()
            loss.backward()
            optimizer_p.step()
            optimizer_s.step()
            optimizer_decoder.step()
            if step%100 == 0:
                print('loss D:%f, loss P:%f loss S:%f' %(lossd.item(), lossp.mean().item(), losss.mean().item()))
                results = tokenizer.batch_decode(predictions)
                results = [tokenizer.convert_tokens_to_string(x) for x in results]
                results = [x.replace(' ', '') for x in results]
                results = [x.replace('[PAD]', '') for x in results]
                print(results)
        test_loss, eval_ans = test(modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, valid_dataloader, loss_func)
        p_eval_loss = test_loss[0]
        s_eval_loss = test_loss[1]
        d_eval_loss = test_loss[2]
        if p_eval_loss > min_loss_p:
            for g in optimizer_p.param_groups:
                g['lr'] = g['lr'] * 0.1
        else:
            min_loss_p = p_eval_loss
        if s_eval_loss > min_loss_s:
            for g in optimizer_s.param_groups:
                g['lr'] = g['lr'] * 0.1
        else:
            min_loss_s = s_eval_loss
        if d_eval_loss < min_loss_d:
            print('New Test Loss:%f' % d_eval_loss)
            t_count = 0
            min_loss_d = d_eval_loss
            #_, _, _, test_loss, result_one = test(model, optimizer_bert, optimizer, test_dataloader, config, record,
            #                                      True)
            state = {'epoch': epoch, 'config': config, 'models': models, 'modelp': modelp, 'model':model, 'eval_rs': eval_ans}
            torch.save(state, './results/'+'best_model.bin')
        else:
            print('New Bigger Test Loss:%f' % d_eval_loss)
            for g in optimizer_decoder.param_groups:
                g['lr'] = g['lr'] * 0.1
    return state

def test(modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, dataloader, loss_func):
    with torch.no_grad():
        modelp.eval()
        models.eval()
        model.eval()
        total_loss = []
        eval_ans = []
        for step, (querys, titles, sections, infer_titles, annotations_ids) in tqdm(enumerate(dataloader)):
            dis_final, lossp = modelp(querys, titles)
            dis_final, losss = models(querys, sections)
            rs2 = modelp.infer(querys, infer_titles)
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
            rs_scores = models.infer(querys, infer_section_candidates_pured)
            scores = scores_title + rs_scores
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
            adj_matrix = get_decoder_att_map(tokenizer, '[SEP]', ids, scores)
            outputs, outputs_logits, pointers = model(ids.cuda(), attention_adjust=adj_matrix,
                                      out_length=2 * annotations_ids['input_ids'].shape[1])
            logits_ = outputs_logits
            targets_ = annotations_ids['input_ids']
            targets_ = batch_pointer_generation(ids, targets_)
            len_anno = min(targets_.shape[1], logits_.shape[1])
            logits = logits_[:, 0:len_anno]
            targets = targets_[:, 0:len_anno]
            pointers = pointers[:, 0:len_anno]
            predictions = batch_pointer_decode(ids, pointers)
            logits = logits.reshape(-1, logits.shape[2])
            targets = targets.reshape(-1).to(config.device)
            results = tokenizer.batch_decode(predictions)
            results = [tokenizer.convert_tokens_to_string(x) for x in results]
            results = [x.replace(' ', '') for x in results]
            results = [x.replace('[PAD]', '') for x in results]
            eval_ans += results
            lossd = loss_func(logits, targets)
            loss = [lossp.mean().item(), losss.mean().item(), lossd.item()]
            total_loss.append(loss)
        modelp.train()
        models.train()
        model.train()
        total_loss = np.array(total_loss).mean(axis=0)
        return total_loss, eval_ans


modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, train_dataloader, valid_dataloader, test_dataloader, loss_func, titles, sections, title2sections, sec2id, bm25_title, bm25_section, tokenizer = build(config)
state = train_eval(modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, train_dataloader, valid_dataloader, loss_func)