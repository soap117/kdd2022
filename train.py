from models.units import MyData, get_decoder_att_map
from config import config
from models.bert_tokenizer import BertTokenizer
from torch.utils.data import DataLoader
from models.retrieval import TitleEncoder, PageRanker, SecEncoder, SectionRanker
from models.modeling_gpt2_att import GPT2LMHeadModel
from tqdm import tqdm
from transformers import AdamW
import numpy as np
import torch
import jieba
def build(config):
    tokenizer = BertTokenizer(vocab_file='./GPT2Chinese/vocab.txt')
    dataset = MyData(config, tokenizer)
    train_dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size
                                  , collate_fn=dataset.collate_fn)
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
    return modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, train_dataloader, dataset, loss_func

def train_eval(modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, train_dataloader, train_dataset, eval_dataset, loss_func):
    min_loss = 1000
    t_count = 0
    for epoch in range(config.train_epoch):
        for step, (querys, titles, sections, infer_titles, annotations_ids) in tqdm(enumerate(train_dataloader)):
            dis_final, lossp = modelp(querys, titles)
            dis_final, losss = models(querys, sections)
            rs2 = modelp.infer(querys, infer_titles)
            rs2 = torch.topk(rs2, 3, dim=1)
            scores_title = rs2[0]
            inds = rs2[1].cpu().numpy()
            infer_title_candidates_pured = []
            infer_section_candidates_pured = []
            mapping_title = np.zeros([len(querys), 3, 4])
            for query, bid in zip(querys, range(len(inds))):
                temp = []
                temp2 = []
                temp3 = []
                for nid, cid in enumerate(inds[bid]):
                    temp.append(infer_titles[bid][cid])
                    temp2 += train_dataset.title2sections[infer_titles[bid][cid]]
                    temp3 += [nid for x in train_dataset.title2sections[infer_titles[bid][cid]]]
                temp2_id = []
                for t_sec in temp2:
                    if t_sec in train_dataset.sec2id:
                        temp2_id.append(train_dataset.sec2id[t_sec])
                key_cut = jieba.lcut(query)
                ls_scores = train_dataset.bm25_section.get_batch_scores(key_cut, temp2_id)
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
            rs_scores = models.infer(querys, infer_section_candidates_pured)
            scores = scores_title * rs_scores
            rs2 = torch.topk(scores, 2, dim=1)
            scores = rs2[0]
            reference = []
            inds_sec = rs2[1].cpu().numpy()
            for bid in range(len(inds_sec)):
                temp = []
                for indc in inds_sec[bid]:
                    temp.append(infer_section_candidates_pured[bid][indc])
                temp = ' [SEP] '.join(temp[0:100])
                reference.append(temp[0:1000])
            inputs = train_dataset.tokenizer(reference, return_tensors="pt", padding=True)
            ids = inputs['input_ids']
            adj_matrix = get_decoder_att_map(train_dataset.tokenizer, '[SEP]', ids, scores)
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
        test_loss = test(modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, train_dataloader, eval_dataset, loss_func)
        if test_loss < min_loss:
            print('New Test Loss:%f' % test_loss)
            t_count = 0
            min_loss = test_loss
            #_, _, _, test_loss, result_one = test(model, optimizer_bert, optimizer, test_dataloader, config, record,
            #                                      True)
            state = {'epoch': epoch, 'config': config, 'models': models, 'modelp':modelp, 'model':model}
            torch.save(state, './results/'+'best_model.bin')
    return state

def test(modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, train_dataloader, train_dataset, loss_func):
    min_loss = 1000
    t_count = 0
    modelp.eval()
    models.eval()
    model.eval()
    total_loss = []
    for step, (querys, titles, sections, infer_titles, annotations_ids) in tqdm(enumerate(train_dataloader)):
        dis_final, lossp = modelp(querys, titles)
        dis_final, losss = models(querys, sections)
        rs2 = modelp.infer(querys, infer_titles)
        rs2 = torch.topk(rs2, 3, dim=1)
        scores_title = rs2[0]
        inds = rs2[1].cpu().numpy()
        infer_title_candidates_pured = []
        infer_section_candidates_pured = []
        mapping_title = np.zeros([len(querys), 3, 4])
        for query, bid in zip(querys, range(len(inds))):
            temp = []
            temp2 = []
            temp3 = []
            for nid, cid in enumerate(inds[bid]):
                temp.append(infer_titles[bid][cid])
                temp2 += train_dataset.title2sections[infer_titles[bid][cid]]
                temp3 += [nid for x in train_dataset.title2sections[infer_titles[bid][cid]]]
            temp2_id = []
            for t_sec in temp2:
                if t_sec in train_dataset.sec2id:
                    temp2_id.append(train_dataset.sec2id[t_sec])
            key_cut = jieba.lcut(query)
            ls_scores = train_dataset.bm25_section.get_batch_scores(key_cut, temp2_id)
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
        rs_scores = models.infer(querys, infer_section_candidates_pured)
        scores = scores_title * rs_scores
        rs2 = torch.topk(scores, 2, dim=1)
        scores = rs2[0]
        reference = []
        inds_sec = rs2[1].cpu().numpy()
        for bid in range(len(inds_sec)):
            temp = []
            for indc in inds_sec[bid]:
                temp.append(infer_section_candidates_pured[bid][indc])
            temp = ' [SEP] '.join(temp[0:100])
            reference.append(temp[0:1000])
        inputs = train_dataset.tokenizer(reference, return_tensors="pt", padding=True)
        ids = inputs['input_ids']
        adj_matrix = get_decoder_att_map(train_dataset.tokenizer, '[SEP]', ids, scores)
        outputs = model(ids.cuda(), attention_adjust=adj_matrix)
        logits = outputs.logits
        targets = annotations_ids['input_ids']
        len_anno = targets.shape[1]
        logits = logits[:, 0:len_anno]
        logits = logits.reshape(-1, logits.shape[2])
        targets = targets.view(-1).to(config.device)
        lossd = loss_func(logits, targets)
        loss = lossp.mean() + losss.mean() + lossd
        total_loss.append(loss.item())
    return total_loss

modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, train_dataloader, dataset, loss_func = build(config)
state = train(modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, train_dataloader, dataset, loss_func)