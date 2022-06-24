import cuda
from eval_units import *
import pickle

import torch
from models.units_sen_editEX import find_spot, operation2sentence_word
from torch.nn.utils.rnn import pad_sequence
from cbert.modeling_cbert import BertForTokenClassification
from transformers import BertTokenizer
from section_inference import preprocess_sec
from cbert.utils.dataset import obtain_indicator
import numpy as np
import jieba
import requests
import time
from models.units import get_decoder_att_map
from config import config
def obtain_annotation(src, tar):
    t_s = 0
    annotations = []
    while t_s < len(tar):
        if tar[t_s] == '（':
            t_e = t_s + 1
            count = 1
            while t_e < len(tar) and count > 0:
                if tar[t_e] == '）':
                    count -= 1
                if tar[t_e] == '（':
                    count += 1
                t_e += 1
            anno_posi = tar[t_s+1:t_e-1]
            if len(anno_posi) > 0 and anno_posi not in src and tar[t_e-1] == '）':
                annotations.append(anno_posi)
            t_s = t_e
        else:
            t_s += 1
    return annotations
from models.retrieval import TitleEncoder, PageRanker, SectionRanker
with open('./data/test/dataset-aligned-para.pkl', 'rb') as f:
    data_test = pickle.load(f)
srcs_ = []
tars_ = []
for point in data_test:
    srcs_.append(point[0])
    tars_.append(point[1])
save_data = torch.load('./results/' + config.data_file.replace('.pkl', '_models_edit_dual_wore.pkl').replace('data/', ''), map_location=config.device)
save_step1_data = torch.load('./cbert/cache/' + 'best_save.data')


bert_model = 'hfl/chinese-bert-wwm-ext'
model_step1 = BertForTokenClassification.from_pretrained(bert_model, num_labels=4)
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
modele = config.modeld_ann.from_pretrained(config.bert_model)
modele.load_state_dict(save_data['modele'])
modele.cuda()
modele.eval()
KEEP_ID = config.tokenizer_editplus.vocab['[unused1]']
DEL_ID = config.tokenizer_editplus.vocab['[unused2]']
INSERT_ID = config.tokenizer_editplus.vocab['[unused5]']
STOP_ID = config.tokenizer_editplus.vocab['[SEP]']
PAD_ID = config.tokenizer_editplus.vocab['[PAD]']
LEFT_ID = config.tokenizer_editplus.vocab['（']
RIGHT_ID = config.tokenizer_editplus.vocab['）']
MARK_ID = config.tokenizer_editplus.vocab['$']
SP_IDS = [KEEP_ID, DEL_ID, INSERT_ID, STOP_ID, PAD_ID, LEFT_ID, RIGHT_ID, MARK_ID]

from models.modeling_bart_ex import BartModel, nn, BartLearnedPositionalEmbedding
from models.modeling_EditNTS_two_rnn_plus import EditDecoderRNN, EditPlus
pos_embed = BartLearnedPositionalEmbedding(1024, 768)
encoder = BartModel.from_pretrained(config.bert_model).encoder
encoder.embed_positions = pos_embed
encoder.embed_tokens = nn.Embedding(config.tokenizer_editplus.vocab_size, config.embedding_new.shape[1], encoder.padding_idx)
encoder.embed_tokens.weight.data[106:] = config.embedding_new[106:]
tokenizer = config.tokenizer_editplus
decoder = EditDecoderRNN(config.tokenizer_editplus.vocab_size, 300, config.rnn_dim, n_layers=config.rnn_layer,
                         embedding=encoder.embed_tokens, SP_IDS=SP_IDS)
edit_nts_ex = EditPlus(encoder, decoder, tokenizer)
modeld = edit_nts_ex
modeld.load_state_dict(save_data['modeld'])
modeld.cuda()
modeld.eval()


def count_score(candidate, reference):
    avg_score = 0
    for k in range(len(candidate)):
        reference_ = reference[k]
        for m in range(len(reference_)):
            reference_[m] = tokenzier_eval.tokenize(reference_[m])
        candidate[k] = tokenzier_eval.tokenize(candidate[k])
        try:
            avg_score += get_sentence_bleu(candidate[k], reference_)/len(candidate)
        except:
            print(candidate[k])
            print(reference[k])
    return avg_score


import re
def is_in_annotation(src, pos):
    s = 0
    count_left = 0
    while s < pos:
        if src[s] == '（':
            count_left += 1
        elif src[s] == '）':
            count_left -= 1
        s += 1
    if count_left > 0:
        return True
    else:
        return False

def fix_stop(tar):
    while re.search(r'（.*(。).*）', tar) is not None:
        tar_stop_list = re.finditer(r'（.*(。).*）', tar)
        for stop in tar_stop_list:
            if is_in_annotation(tar, stop.regs[1][0]):
                temp = list(tar)
                temp[stop.regs[1][0]] = '\\'
                tar = ''.join(temp)
            else:
                temp = list(tar)
                temp[stop.regs[1][0]] = '\n'
                tar = ''.join(temp)
    tar = tar.replace('\\', '')
    tar = tar.replace('\n', '。')
    return tar
import copy
import json
def pipieline(path_from):
    eval_ans = []
    eval_gt = []
    record_scores = []
    record_references = []

    srcs = []
    tars = []
    for src, tar in zip(srcs_, tars_):
        src = re.sub('\*\*', '', src)
        src = src.replace('(', '（')
        src = src.replace('$', '')
        src = src.replace(')', '）')
        src = src.replace('\n', '').replace('。。', '。')
        src = fix_stop(src)
        tar = re.sub('\*\*', '', tar)
        tar = tar.replace('\n', '').replace('。。', '。')
        tar = tar.replace('(', '（')
        tar = tar.replace(')', '）')
        tar = tar.replace('$', '')
        tar = fix_stop(tar)
        if src[-1] == '。' and tar[-1] != '。':
            tar += '。'
        if tar[-1] == '。' and src[-1] != '。':
            src += '。'
        srcs.append(src)
        tars.append(tar)

    for src, tar in zip(srcs, tars):
        src_ori = copy.copy(src)
        decoder_inputs = tokenizer([src], return_tensors="pt", padding=True, truncation=True)
        decoder_anno_position = []
        hidden_annotation = None
        decoder_ids = decoder_inputs['input_ids']
        edit_sens = [['[SEP]']]
        edit_sens_token = [['[CLS]'] + x + ['[SEP]'] for x in edit_sens]
        edit_sens_token_ids = [torch.LongTensor(tokenizer.convert_tokens_to_ids(x)) for x in edit_sens_token]
        edit_sens_token_ids = pad_sequence(edit_sens_token_ids, batch_first=True, padding_value=0).to(config.device)
        input_actions = torch.zeros_like(edit_sens_token_ids) + 5
        input_actions = torch.where(
            (edit_sens_token_ids == 1) | (edit_sens_token_ids == 2) | (edit_sens_token_ids == 101) | (
                    edit_sens_token_ids == 102), edit_sens_token_ids,
            input_actions)
        clean_indication = None
        decoder_ids = decoder_ids.to(config.device)
        target_ids = tokenizer([src], return_tensors="pt", padding=True, truncation=True)['input_ids'].to(
            config.device)
        logits_action, logits_edit, hidden_edits = modeld(input_ids=decoder_ids, decoder_input_ids=target_ids,
                                                          anno_position=decoder_anno_position,
                                                          hidden_annotation=hidden_annotation,
                                                          input_edits=edit_sens_token_ids,
                                                          input_actions=input_actions, org_ids=None,
                                                          force_ratio=0.0, clean_indication=clean_indication)

        _, action_predictions = torch.max(logits_action, dim=-1)
        _, edit_predictions = torch.max(logits_edit, dim=-1)
        predictions = torch.where(action_predictions != 5, action_predictions, edit_predictions)

        decoder_inputs = tokenizer([src], return_tensors="pt", padding=True)
        decoder_ids = decoder_inputs['input_ids']
        predictions, predictions_text = operation2sentence_word(predictions, decoder_ids, tokenizer.tokenize([src]), tokenizer)
        results = predictions_text
        results = [x.replace(' ', '') for x in results]
        results = [x.replace('[PAD]', '') for x in results]
        results = [x.replace('[CLS]', '') for x in results]
        results = [x.replace('[MASK]', '').replace('[unused3]', '').replace('[unused4]', '') for x in results]
        results = [x.split('[SEP]')[0] for x in results]
        results = [x.replace('（）', '') for x in results]
        results = [x.replace('$', '') for x in results]
        p_annos = obtain_annotation(results[1], results[0])
        if len(p_annos)==0:
            results[0] = results[1]
            print("skip useless modify")
        else:
            print('+++++++++++++++++++++++')
            print(results[0])
            print('+++++++++++++++++++++++')
        # masks = torch.ones_like(targets)
        # masks[torch.where(targets == 0)] = 0
        eval_ans += [results[0]]
        eval_gt += [tar]
        #print(eval_ans[-1])

    result_final = {'srcs': srcs, 'prds': eval_ans, 'tars': eval_gt, 'scores': record_scores,
                    'reference': record_references}
    with open('./data/test/my_results_edit_para_dual_wo.pkl', 'wb') as f:
        pickle.dump(result_final, f)




pipieline('./data/test')
