import cuda
import torch
import torch.nn as nn
from config import Config
config = Config(1)
from models.units_sen_editEX import MyData, get_decoder_att_map, mask_ref, read_clean_data, find_spot, restricted_decoding, operation2sentence
from torch.utils.data import DataLoader
from models.retrieval import TitleEncoder, PageRanker, SecEncoder, SectionRanker
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from transformers import AdamW
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
smooth = SmoothingFunction()
import jieba
from rank_bm25 import BM25Okapi
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
    debug_flag = False
    if not debug_flag and os.path.exists(config.data_file.replace('.pkl', '_train_dataset_edit.pkl')):
        train_dataset = torch.load(config.data_file.replace('.pkl', '_train_dataset_edit.pkl'))
        valid_dataset = torch.load(config.data_file.replace('.pkl', '_valid_dataset_edit.pkl'))
        test_dataset = torch.load(config.data_file.replace('.pkl', '_test_dataset_edit.pkl'))
    else:
        train_dataset = MyData(config, tokenizer, config.data_file.replace('.pkl', '_train_dataset_raw.pkl'), titles, sections, title2sections, sec2id, bm25_title, bm25_section)
        valid_dataset = MyData(config, tokenizer, config.data_file.replace('.pkl', '_valid_dataset_raw.pkl'), titles, sections, title2sections, sec2id,
                               bm25_title,
                               bm25_section)
        test_dataset = MyData(config, tokenizer, config.data_file.replace('.pkl', '_test_dataset_raw.pkl'), titles, sections, title2sections, sec2id, bm25_title,
                              bm25_section)
        torch.save(train_dataset, config.data_file.replace('.pkl', '_train_dataset_edit.pkl'))
        torch.save(valid_dataset, config.data_file.replace('.pkl', '_valid_dataset_edit.pkl'))
        torch.save(test_dataset, config.data_file.replace('.pkl', '_test_dataset_edit.pkl'))

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size
                                  , collate_fn=train_dataset.collate_fn)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=config.batch_size
                                  , collate_fn=train_dataset.collate_fn_test)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=config.batch_size
                                  , collate_fn=train_dataset.collate_fn_test)

    title_encoder = TitleEncoder(config)
    save_data = torch.load('./results/' + config.data_file_anno.replace('.pkl', '_models_full.pkl').replace('data/', ''))
    modelp = PageRanker(config, title_encoder)
    print('Load pretrained P')
    modelp.load_state_dict(save_data['modelp'])
    models = SectionRanker(config, title_encoder)
    models.load_state_dict(save_data['models'])
    print('Load pretrained S')
    modele = config.modeld_ann.from_pretrained(config.bert_model)
    modele.load_state_dict(save_data['model'])
    print('Load pretrained E')
    from transformers import BertTokenizer
    from models.modeling_bart_ex import BartModel
    from models.modeling_EditNTS_plus import EditDecoderRNN, EditPlus
    bert_model = config.bert_model
    encoder = BartModel.from_pretrained(bert_model).encoder
    tokenizer = config.tokenizer
    decoder = EditDecoderRNN(tokenizer.vocab_size, 768, 400, n_layers=1, embedding=encoder.embed_tokens)
    edit_nts_ex = EditPlus(encoder, decoder, tokenizer)
    modeld = edit_nts_ex

    save_data = torch.load('./results/' + config.data_file.replace('.pkl', '_models_edit.pkl').replace('data/', ''))
    modeld.load_state_dict(save_data['modeld'])
    modeld.cuda()
    modelp.load_state_dict(save_data['modelp'])
    modelp.cuda()
    models.load_state_dict(save_data['models'])
    models.cuda()
    modele.load_state_dict(save_data['modele'])
    modele.cuda()
    modelp.train()
    models.train()
    modele.train()
    modeld.train()
    optimizer_p = AdamW(modelp.parameters(), lr=config.lr*0.1)
    optimizer_s = AdamW(models.parameters(), lr=config.lr*0.1)
    optimizer_encoder = AdamW(modele.parameters(), lr=config.lr*0.01)
    optimizer_decoder = AdamW(modeld.parameters(), lr=config.lr*0.1)
    loss_func = nn.NLLLoss(ignore_index=config.tokenizer.vocab['[PAD]'], reduction='none')
    return modelp, models, modele, modeld, optimizer_p, optimizer_s, optimizer_encoder, optimizer_decoder, train_dataloader, valid_dataloader, test_dataloader, loss_func, titles, sections, title2sections, sec2id, bm25_title, bm25_section, tokenizer

def train_eval(modelp, models, modele, modeld, optimizer_p, optimizer_s, optimizer_encoder, optimizer_decoder, train_dataloader, valid_dataloader, loss_func):
    min_loss_p = min_loss_s = min_loss_d = 1000
    state = {}
    count_s = -1
    count_p = -1
    data_size = len(train_dataloader)
    test_loss, eval_ans, grand_ans = test(modelp, models, modele, modeld, valid_dataloader, loss_func)
    return state



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
        for step, (querys, querys_ori, querys_context, titles, sections, infer_titles, src_sens, src_sens_ori, tar_sens, cut_list, pos_titles, pos_sections, edit_sens) in zip(
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

            an_decoder_input = ' '.join(['[MASK]' for x in range(100)])
            an_decoder_inputs = [an_decoder_input for x in reference_ids]
            an_decoder_inputs = tokenizer(an_decoder_inputs, return_tensors="pt", padding=True)
            an_decoder_inputs_ids = an_decoder_inputs['input_ids'].to(config.device)

            decoder_inputs = tokenizer(src_sens, return_tensors="pt", padding=True, truncation=True)
            decoder_ids = decoder_inputs['input_ids']
            decoder_inputs_ori = tokenizer(src_sens_ori, return_tensors="pt", padding=True, truncation=True)
            decoder_ids_ori = decoder_inputs_ori['input_ids'].to(config.device)

            edit_sens_token = [['[CLS]'] + x + ['[SEP]'] for x in edit_sens]
            edit_sens_token_ids = [torch.LongTensor(tokenizer.convert_tokens_to_ids(x)) for x in edit_sens_token]
            edit_sens_token_ids = pad_sequence(edit_sens_token_ids, batch_first=True, padding_value=0).to(config.device)
            #decoder_edits = tokenizer(edit_sens.replace(' ', ''), return_tensors="pt", padding=True, truncation=True)
            #decoder_ids_edits = decoder_edits['input_ids'].to(config.device)

            decoder_anno_position = find_spot(decoder_ids, querys_ori, tokenizer)
            decoder_ids = decoder_ids.to(config.device)
            target_ids = tokenizer(tar_sens, return_tensors="pt", padding=True, truncation=True)['input_ids'].to(
                config.device)
            adj_matrix = get_decoder_att_map(tokenizer, '[SEP]', reference_ids, scores)

            outputs_annotation = modele(input_ids=reference_ids, attention_adjust=adj_matrix, decoder_input_ids=an_decoder_inputs_ids)
            hidden_annotation = outputs_annotation.decoder_hidden_states[:, 0:config.hidden_anno_len]

            logits, hidden_edits = modeld(input_ids=decoder_ids, decoder_input_ids=target_ids,
                                          anno_position=decoder_anno_position, hidden_annotation=hidden_annotation,
                                          input_edits=edit_sens_token_ids, org_ids=decoder_ids_ori, force_ratio=0.0)

            targets = target_ids[:, 1:]
            _, predictions = torch.max(logits, dim=-1)
            print(predictions)
            predictions = operation2sentence(predictions, decoder_ids)
            results = tokenizer.batch_decode(predictions)
            results = [tokenizer.convert_tokens_to_string(x) for x in results]
            results = [x.replace(' ', '') for x in results]
            results = [x.replace('[PAD]', '') for x in results]
            results = [x.replace('[CLS]', '') for x in results]
            results = [x.replace('[MASK]', '') for x in results]
            results = [x.split('[SEP]')[0] for x in results]
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