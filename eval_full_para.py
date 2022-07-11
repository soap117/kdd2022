import torch
import torch.nn as nn
from config import Config
config = Config(8)
from models.units_sen import MyData, get_decoder_att_map, mask_ref
from torch.utils.data import DataLoader
from models.retrieval import TitleEncoder, PageRanker, SecEncoder, SectionRanker
from tqdm import tqdm
from transformers import AdamW
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
smooth = SmoothingFunction()
import jieba
from models.units_sen import read_clean_data, find_spot, restricted_decoding
from rank_bm25 import BM25Okapi
from models.modeling_gpt2_att import GPT2LMHeadModel
from models.modeling_bart_att import BartForConditionalGeneration
import os
class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        return getattr(self.module, name)
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
                                  , collate_fn=train_dataset.collate_fn_test)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=config.batch_size
                                  , collate_fn=train_dataset.collate_fn_test)

    title_encoder = TitleEncoder(config)
    save_data = torch.load('./results/' + config.data_file.replace('.pkl', '_models_full.pkl').replace('data/', ''))
    modelp = PageRanker(config, title_encoder)
    print('Load pretrained P')
    modelp.load_state_dict(save_data['modelp'])
    models = SectionRanker(config, title_encoder)
    models.load_state_dict(save_data['models'])
    print('Load pretrained S')
    modele = config.modeld_ann.from_pretrained(config.bert_model)
    modele.load_state_dict(save_data['modele'])
    print('Load pretrained E')
    modeld = config.modeld_sen.from_pretrained(config.bert_model)
    modeld.load_state_dict(save_data['modeld'])
    modeld.cuda()
    modelp.cuda()
    models.cuda()
    modele.cuda()
    optimizer_p = AdamW(modelp.parameters(), lr=config.lr*0.1)
    optimizer_s = AdamW(models.parameters(), lr=config.lr*0.1)
    optimizer_encoder = AdamW(modele.parameters(), lr=config.lr * 0.1)
    optimizer_decoder = AdamW(modeld.parameters(), lr=config.lr * 0.1)
    loss_func = torch.nn.CrossEntropyLoss(reduction='none')
    return modelp, models, modele, modeld, optimizer_p, optimizer_s, optimizer_encoder, optimizer_decoder, train_dataloader, valid_dataloader, test_dataloader, loss_func, titles, sections, title2sections, sec2id, bm25_title, bm25_section, tokenizer

def train_eval(modelp, models, modele, modeld, optimizer_p, optimizer_s, optimizer_encoder, optimizer_decoder, train_dataloader, valid_dataloader, loss_func):
    min_loss_p = min_loss_s = min_loss_d = 1000
    state = {}
    count_s = -1
    count_p = -1
    data_size = len(train_dataloader)
    test_loss, eval_ans, eval_grd = test(modelp, models, modele, modeld, valid_dataloader, loss_func)



def test(modelp, models, modele, modeld, dataloader, loss_func):
    with torch.no_grad():
        modelp.eval()
        models.eval()
        modele.eval()
        modeld.eval()
        total_loss = []
        tp = 0
        total = 0
        tp_s = 0
        total_s = 0
        eval_ans = []
        eval_gt = []
        data_size = len(dataloader)
        for step, (querys, querys_ori, querys_context, titles, sections, infer_titles, src_sens, tar_sens, cut_list, pos_titles, pos_sections) in zip(
                tqdm(range(data_size)), dataloader):
            dis_final, lossp, query_embedding = modelp(querys, querys_context, titles)
            rs2 = modelp(query_embedding=query_embedding, candidates=infer_titles, is_infer=True)
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
            rs_scores = models(query_embedding, infer_section_candidates_pured, is_infer=True)
            scores = scores_title * rs_scores
            rs2 = torch.topk(scores, config.infer_section_select, dim=1)
            scores = rs2[0]
            reference = []
            inds_sec = rs2[1].cpu().numpy()
            for bid in range(len(inds_sec)):
                total_s += 1
                temp = [querys[bid]]
                for indc in inds_sec[bid]:
                    temp.append(infer_section_candidates_pured[bid][indc][0:100])
                if check(query, temp, pos_sections[bid], secs=True):
                    tp_s += 1
                temp = ' [SEP] '.join(temp)
                reference.append(temp[0:config.maxium_sec])
            inputs_ref = tokenizer(reference, return_tensors="pt", padding=True, truncation=True)
            reference_ids = inputs_ref['input_ids'].to(config.device)
            adj_matrix = get_decoder_att_map(tokenizer, '[SEP]', reference_ids, scores)
            outputs_annotation = modele(input_ids=reference_ids, attention_adjust=adj_matrix)
            hidden_annotation = outputs_annotation.decoder_hidden_states[:, 0:config.hidden_anno_len]
            results, target_ids = restricted_decoding(querys_ori, src_sens, tar_sens, hidden_annotation, tokenizer, modeld)
            targets = target_ids[:, 1:]
            ground_truth = tokenizer.batch_decode(targets)
            ground_truth = [tokenizer.convert_tokens_to_string(x) for x in ground_truth]
            ground_truth = [x.replace(' ', '') for x in ground_truth]
            ground_truth = [x.replace('[PAD]', '') for x in ground_truth]
            ground_truth = [x.replace('[CLS]', '') for x in ground_truth]
            ground_truth = [x.split('[SEP]')[0] for x in ground_truth]
            #masks = torch.ones_like(targets)
            #masks[torch.where(targets == 0)] = 0
            eval_ans += results
            eval_gt += ground_truth
        predictions = [tokenizer.tokenize(doc) for doc in eval_ans]
        reference = [[tokenizer.tokenize(doc)] for doc in eval_gt]
        bleu_scores = corpus_bleu(reference, predictions,)
        print("Bleu Annotation:%f" % bleu_scores)
        modelp.train()
        models.train()
        modele.train()
        modeld.train()
        print('accuracy title: %f accuracy section: %f' % (tp / total, tp_s / total_s))
        return (-tp / total, -tp_s / total_s, -bleu_scores), eval_ans, eval_gt

modelp, models, modele, modeld, optimizer_p, optimizer_s, optimizer_encoder, optimizer_decoder, train_dataloader, valid_dataloader, test_dataloader, loss_func, titles, sections, title2sections, sec2id, bm25_title, bm25_section, tokenizer = build(config)
state = train_eval(modelp, models, modele, modeld, optimizer_p, optimizer_s, optimizer_encoder, optimizer_decoder, train_dataloader, valid_dataloader, loss_func)