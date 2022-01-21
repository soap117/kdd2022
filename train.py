import torch
from config import config
from models.units import MyData, get_decoder_att_map
from models.bert_tokenizer import BertTokenizer
from torch.utils.data import DataLoader
from models.retrieval import TitleEncoder, PageRanker, SecEncoder, SectionRanker
from models.modeling_gpt2_att import GPT2LMHeadModel
from tqdm import tqdm
from transformers import AdamW
import numpy as np
import jieba
from models.units import read_clean_data
from rank_bm25 import BM25Okapi
def build(config):
    tokenizer = BertTokenizer(vocab_file='./GPT2Chinese/vocab.txt', do_lower_case=False, never_split=['[SEP]'])
    titles, sections, title2sections, sec2id = read_clean_data('data/mydata_new_baidu.pkl')
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
    model = GPT2LMHeadModel.from_pretrained("./GPT2Chinese/")
    model.train()
    model.cuda()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer_p = AdamW(modelp.parameters(), lr=config.lr)
    optimizer_s = AdamW(models.parameters(), lr=config.lr)
    optimizer_decoder = AdamW(optimizer_grouped_parameters, lr=config.lr)
    loss_func = torch.nn.CrossEntropyLoss()
    return modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, train_dataloader, valid_dataloader, test_dataloader, loss_func, titles, sections, title2sections, sec2id, bm25_title, bm25_section, tokenizer

def train_eval(modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, train_dataloader, valid_dataloader, loss_func):
    min_loss = 1000
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
                temp3_pured = []
                for oid, one in enumerate(cindex):
                    temp2_pured.append(temp2[one])
                    temp3_pured.append(temp3[one])
                    mapping_title[bid, temp3[one], oid] = 1
                while len(temp2_pured) < config.infer_section_range:
                    temp2_pured.append('')
                    temp3_pured.append(-1)

                infer_title_candidates_pured.append(temp)
                infer_section_candidates_pured.append(temp2_pured)

            mapping = torch.FloatTensor(mapping_title).to(config.device)
            scores_title = scores_title.unsqueeze(1)
            scores_title = scores_title.matmul(mapping).squeeze(1)
            rs_scores = models.infer(querys, infer_section_candidates_pured)
            scores = scores_title * rs_scores
            rs2 = torch.topk(scores, config.infer_section_select, dim=1)
            scores = rs2[0]
            reference = []
            inds_sec = rs2[1].cpu().numpy()
            for bid in range(len(inds_sec)):
                temp = []
                for indc in inds_sec[bid]:
                    temp.append(infer_section_candidates_pured[bid][indc])
                temp = ' [SEP] '.join(temp[0:300])
                reference.append(temp[0:1000])
            inputs = tokenizer(reference, return_tensors="pt", padding=True)
            ids = inputs['input_ids']
            adj_matrix = get_decoder_att_map(tokenizer, '[SEP]', ids, scores)
            outputs = model(ids.cuda(), attention_adjust=adj_matrix)
            logits = outputs.logits
            targets = annotations_ids['input_ids']
            len_anno = targets.shape[1]
            logits = logits[:, 0:len_anno]
            logits = logits.reshape(-1, logits.shape[2])
            targets = targets.view(-1).to(config.device)
            lossd = loss_func(logits, targets)
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
                _, predictions = torch.max(logits, dim=-1)
                results = tokenizer.batch_decode(predictions)
                results = tokenizer.convert_tokens_to_string(results)
                results = [x.replace('[PAD]', '') for x in results]
                print(results)
        test_loss, eval_ans = test(modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, valid_dataloader, loss_func)
        if test_loss < min_loss:
            print('New Test Loss:%f' % test_loss)
            t_count = 0
            min_loss = test_loss
            #_, _, _, test_loss, result_one = test(model, optimizer_bert, optimizer, test_dataloader, config, record,
            #                                      True)
            state = {'epoch': epoch, 'config': config, 'models': models, 'modelp':modelp, 'model':model}
            torch.save(state, './results/'+'best_model.bin')
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
                temp3_pured = []
                for oid, one in enumerate(cindex):
                    temp2_pured.append(temp2[one])
                    temp3_pured.append(temp3[one])
                    mapping_title[bid, temp3[one], oid] = 1
                while len(temp2_pured) < config.infer_section_range:
                    temp2_pured.append('')
                    temp3_pured.append(-1)

                infer_title_candidates_pured.append(temp)
                infer_section_candidates_pured.append(temp2_pured)

            mapping = torch.FloatTensor(mapping_title).to(config.device)
            scores_title = scores_title.unsqueeze(1)
            scores_title = scores_title.matmul(mapping).squeeze(1)
            rs_scores = models.infer(querys, infer_section_candidates_pured)
            scores = scores_title * rs_scores
            rs2 = torch.topk(scores, config.infer_section_select, dim=1)
            scores = rs2[0]
            reference = []
            inds_sec = rs2[1].cpu().numpy()
            for bid in range(len(inds_sec)):
                temp = []
                for indc in inds_sec[bid]:
                    temp.append(infer_section_candidates_pured[bid][indc])
                temp = ' [SEP] '.join(temp[0:200])
                reference.append(temp[0:1000])
            inputs = tokenizer(reference, return_tensors="pt", padding=True)
            ids = inputs['input_ids']
            adj_matrix = get_decoder_att_map(tokenizer, '[SEP]', ids, scores)
            outputs = model(ids.cuda(), attention_adjust=adj_matrix)
            logits = outputs.logits
            _, predictions = torch.max(logits, dim=-1)
            results = tokenizer.batch_decode(predictions)
            results = tokenizer.convert_tokens_to_string(results)
            results = [x.replace('[PAD]', '') for x in results]
            eval_ans += results
            targets = annotations_ids['input_ids']
            len_anno = targets.shape[1]
            logits = logits[:, 0:len_anno]
            logits = logits.reshape(-1, logits.shape[2])
            targets = targets.view(-1).to(config.device)
            lossd = loss_func(logits, targets)
            loss = lossp.mean() + losss.mean() + lossd
            total_loss.append(loss.item())
        modelp.train()
        models.train()
        model.train()
        return np.array(total_loss).mean(), eval_ans


modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, train_dataloader, valid_dataloader, test_dataloader, loss_func, titles, sections, title2sections, sec2id, bm25_title, bm25_section, tokenizer = build(config)
state = train_eval(modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, train_dataloader, valid_dataloader, loss_func)