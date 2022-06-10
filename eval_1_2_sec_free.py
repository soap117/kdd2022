import cuda2
from eval_units import *
import pickle

import torch
from models.units_sen import restricted_decoding
from cbert.modeling_cbert import BertForTokenClassification
from transformers import BertTokenizer
from section_inference import preprocess_sec
from cbert.utils.dataset import obtain_indicator
import numpy as np
batch_size = 4
import jieba
import requests
import time
from models.units import get_decoder_att_map
from config import config
from models.retrieval import TitleEncoder, PageRanker, SectionRanker
with open('./data/test/dataset-aligned-para.pkl', 'rb') as f:
    data_test = pickle.load(f)
srcs_ = []
tars_ = []
for point in data_test:
    srcs_.append(point[0])
    tars_.append(point[1])
save_data = torch.load('./results/' + config.data_file.replace('.pkl', '_models_full.pkl').replace('data/', ''))
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
modeld = config.modeld_sen.from_pretrained(config.bert_model)
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

def obtain_step2_input(pre_labels, src, src_ids, step1_tokenizer):
    input_list = [[],[],[], [],[]]
    l = 0
    r = 0
    while src_ids[r] != step1_tokenizer.vocab['。']:
        r += 1
    for c_id in range(len(src_ids)):
        if src_ids[c_id] == step1_tokenizer.vocab['。']:
            context = step1_tokenizer.decode(src_ids[l:r+1]).replace(' ', '').replace('[CLS]', '').replace('[SEP]', '')
            input_list[4].append((False, context))
            l = r + 1
            r = l+1
            while r<len(src_ids) and src_ids[r] != step1_tokenizer.vocab['。']:
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
            context = step1_tokenizer.decode(src_ids[l:r+1]).replace(' ', '').replace('[CLS]', '').replace('[SEP]', '')
            key_cut = jieba.lcut(key)
            infer_titles = bm25_title.get_top_n(key_cut, titles, config.infer_title_range)
            if len(key) > 0:
                input_list[0].append(key)
                input_list[1].append(context)
                input_list[2].append(infer_titles)
                input_list[3].append((l_k, r_k))
    return input_list

def check_seq(a, b):
    for x_a, x_b in zip(a, b):
        if x_a != x_b:
            return False
    return True

def find_spot(input_ids, querys_ori, tokenizer):
    positions = []
    for ori_query in querys_ori:
        flag = False
        format = '${}$'.format(ori_query)
        format_id = tokenizer(format)['input_ids'][1:-1]
        for bid in range(input_ids.shape[0]):
            l = 0
            while input_ids[bid, l] != config.SEP and not check_seq(input_ids[bid, l:l+len(format_id)], format_id):
               l += 1
            if input_ids[bid, l] != config.SEP:
                positions.append((bid, l+len(format_id), l+len(format_id)+10))
                flag = True
                break
        if not flag:
            positions.append((-1,-1,-1))
    return positions

def mark_sentence(input_list):
    context_dic = {}
    for key, context, infer_titles in zip(input_list[0], input_list[1], input_list[2]):
        if context not in context_dic:
            src = context
            src = re.sub('\*\*', '', src)
            src = src.replace('(', '（')
            src = src.replace('$', '')
            src = src.replace(')', '）')
            src = src.replace('\n', '').replace('。。', '。')
            src = fix_stop(src)
            context_dic[context] = [src, src, [], []]
            src_sentence = context_dic[context][0]
            tar_sentence = context_dic[context][1]

        else:
            src_sentence = context_dic[context][0]
            tar_sentence = context_dic[context][1]
        context_dic[context][2].append(key)
        context_dic[context][3].append(infer_titles)
        region = re.search(key, src_sentence)
        if region is not None:
            region = region.regs[0]
        else:
            region = (0, 0)
        if region[0] != 0 or region[1] != 0:
            src_sentence = src_sentence[0:region[0]] + ' ${}$ '.format(key) + ''.join(
                [' [MASK] ' for x in range(10)]) + src_sentence[region[1]:]
        region = re.search(key, tar_sentence)
        if region is not None:
            region = region.regs[0]
        else:
            region = (0, 0)
        if region[0] != 0 or region[1] != 0:
            tar_sentence = tar_sentence[0:region[0]] + ' ${}$ （）'.format(key) + tar_sentence[region[1]:]
            '''
            if region[1] < len(tar_sentence) and tar_sentence[region[1]] != '（' and region[1] + 1 < len(
                    tar_sentence) and tar_sentence[region[1] + 1] != '（' or region[1] == len(tar_sentence) or region[1]+1 == len(tar_sentence):
                tar_sentence = tar_sentence[0:region[0]] + ' ${}$ （）'.format(key) + tar_sentence[region[1]:]
            else:
                tar_sentence = tar_sentence[0:region[0]] + ' ${}$ '.format(key) + tar_sentence[region[1]:]
            '''
        context_dic[context][0] = src_sentence
        context_dic[context][1] = tar_sentence
    order_context = []
    for context in input_list[4]:
        if context[1] not in order_context:
            order_context.append(context[1])
    return context_dic, order_context

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
    tokenizer = config.tokenizer

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
        context_dic, order_context = mark_sentence(step2_input)
        batch_rs = {}
        for context in context_dic.keys():
            querys = context_dic[context][2]
            querys_ori = copy.copy(querys)
            src = context_dic[context][0]
            src_tar = context_dic[context][1]
            infer_titles = context_dic[context][3]
            temp = []
            for query in querys:
                if query in mark_key_equal:
                    temp.append(mark_key_equal[query])
                else:
                    url = 'https://api.ownthink.com/kg/ambiguous?mention=%s' % query
                    headers = {
                        'User-Agent': 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
                    }
                    try_count = 0
                    while try_count < 3:
                        try:
                            r = requests.get(url, headers=headers, timeout=5)
                            break
                        except Exception as e:
                            try_count += 1
                            print("trying %d time" % try_count)
                            wait_gap = 3
                            time.sleep((try_count + np.random.rand()) * wait_gap)
                    rs = json.loads(r.text)
                    rs = rs['data']
                    name_set = set()
                    name_set.add(query)
                    for one in rs:
                        name_set.add(re.sub(r'\[.*\]', '', one[0]))
                    name_list = list(name_set)
                    new_query = ' '.join(name_list)
                    if len(new_query) == 0:
                        new_query = query
                    if query != new_query:
                        print("%s->%s" % (query, new_query))
                    mark_key_equal[query] = new_query
                    temp.append(new_query)
            querys = temp
            if len(querys) == 0:
                continue
            contexts = []
            for query in querys:
                contexts.append(context)
            query_embedding = modelp.query_embeddings(querys, contexts)
            dis_scores = modelp(query_embedding=query_embedding, candidates=infer_titles, is_infer=True)
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
            record_scores.append(scores.detach().cpu().numpy())
            record_references.append(reference)
            inputs_ref = tokenizer(reference, return_tensors="pt", padding=True, truncation=True)
            reference_ids = inputs_ref['input_ids'].to(config.device)
            adj_matrix = get_decoder_att_map(tokenizer, '[SEP]', reference_ids, scores)

            an_decoder_input = ' '.join(['[MASK]' for x in range(100)])
            an_decoder_inputs = [an_decoder_input for x in reference_ids]
            an_decoder_inputs = tokenizer(an_decoder_inputs, return_tensors="pt", padding=True)
            an_decoder_inputs_ids = an_decoder_inputs['input_ids'].to(config.device)

            outputs_annotation = modele(input_ids=reference_ids, attention_adjust=adj_matrix,
                                        decoder_input_ids=an_decoder_inputs_ids)

            hidden_annotation = outputs_annotation.decoder_hidden_states[:, 0:config.hidden_anno_len]
            results, target_ids = restricted_decoding(querys_ori, [src], [src_tar], hidden_annotation, tokenizer, modeld, is_free=True)
            results = [x.replace('（）', '') for x in results]
            print(results[0])
            results = [x.replace('$', '') for x in results]
            # masks = torch.ones_like(targets)
            # masks[torch.where(targets == 0)] = 0
            batch_rs[context] = results[0]
        section_rs = []
        for context in order_context:
            if context in batch_rs:
                section_rs += [batch_rs[context]]
            else:
                section_rs += [context]
        section_rs = '。'.join(section_rs)
        section_rs += '。'
        eval_gt += [tar]
        eval_ans += [section_rs]

    result_final = {'srcs': srcs, 'prds': eval_ans, 'tars': eval_gt, 'scores': record_scores,
                    'reference': record_references}
    with open('./data/test/my_results_sec.pkl', 'wb') as f:
        pickle.dump(result_final, f)




pipieline('./data/test')
