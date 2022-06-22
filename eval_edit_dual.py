import cuda
from eval_units import *
import pickle

import torch
from models.units_sen_editEX import find_spot, operation2sentence
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
save_data = torch.load('./results/' + config.data_file_old.replace('.pkl', '_models_edit_dual_plus_rn_%d_%d.pkl' %(config.rnn_dim, config.rnn_layer)).replace('data/', ''))
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
from models.modeling_bart_ex import BartModel
from models.modeling_EditNTS_two_rnn_plus import EditDecoderRNN, EditPlus
bert_model = config.bert_model
encoder = BartModel.from_pretrained(bert_model).encoder
tokenizer = config.tokenizer
decoder = EditDecoderRNN(tokenizer.vocab_size, 768, config.rnn_dim, n_layers=config.rnn_layer, embedding=encoder.embed_tokens)
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
def pre_process_sentence(src, tar, keys):
    src = '。'.join(src)
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
    src_sentence = src.split('。')
    tar_sentence = tar.split('。')
    for key in keys:
        region = re.search(key, src)
        if region is not None:
            region = region.regs[0]
        else:
            region = (0, 0)
        if region[0] != 0 or region[1] != 0:
            src_sentence = src_sentence[0:region[0]] + ' ${}$ '.format(key) + ''.join(
                [' [MASK] ' for x in range(config.hidden_anno_len_rnn)]) + src_sentence[region[1]:]
        region = re.search(key, tar_sentence)
        if region is not None:
            region = region.regs[0]
        else:
            region = (0, 0)
        if region[0] != 0 or region[1] != 0:
            if region[1] < len(tar_sentence) and tar_sentence[region[1]] != '（' and region[1] + 1 < len(
                    tar_sentence) and tar_sentence[region[1] + 1] != '（' and region[1] + 2 < len(tar_sentence) and \
                    tar_sentence[region[1] + 2] != '（':
                tar_sentence = tar_sentence[0:region[0]] + ' ${}$ （）'.format(key) + tar_sentence[region[1]:]
            else:
                tar_sentence = tar_sentence[0:region[0]] + ' ${}$ '.format(key) + tar_sentence[region[1]:]
    src = src_sentence
    return src, tar_sentence, tar
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
        context_dic, order_context = mark_sentence_rnn(step2_input)
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

            an_decoder_input = ' '.join(['[MASK]' for x in range(100)])
            an_decoder_inputs = [an_decoder_input for x in reference_ids]
            an_decoder_inputs = tokenizer(an_decoder_inputs, return_tensors="pt", padding=True)
            an_decoder_inputs_ids = an_decoder_inputs['input_ids'].to(config.device)

            decoder_inputs = tokenizer([src], return_tensors="pt", padding=True, truncation=True)
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
            # decoder_edits = tokenizer(edit_sens.replace(' ', ''), return_tensors="pt", padding=True, truncation=True)
            # decoder_ids_edits = decoder_edits['input_ids'].to(config.device)

            decoder_anno_position = find_spot(decoder_ids, querys_ori, tokenizer)
            decoder_ids = decoder_ids.to(config.device)
            target_ids = tokenizer([src_tar], return_tensors="pt", padding=True, truncation=True)['input_ids'].to(
                config.device)
            adj_matrix = get_decoder_att_map(tokenizer, '[SEP]', reference_ids, scores)

            outputs_annotation = modele(input_ids=reference_ids, attention_adjust=adj_matrix,
                                        decoder_input_ids=an_decoder_inputs_ids)
            hidden_annotation = outputs_annotation.decoder_hidden_states[:, 0:config.hidden_anno_len_rnn]

            logits_action, logits_edit, hidden_edits = modeld(input_ids=decoder_ids, decoder_input_ids=target_ids,
                                                              anno_position=decoder_anno_position,
                                                              hidden_annotation=hidden_annotation,
                                                              input_edits=edit_sens_token_ids,
                                                              input_actions=input_actions, org_ids=None,
                                                              force_ratio=0.0, eval=True)

            _, action_predictions = torch.max(logits_action, dim=-1)
            _, edit_predictions = torch.max(logits_edit, dim=-1)
            predictions = torch.where(action_predictions != 5, action_predictions, edit_predictions)

            predictions = operation2sentence(predictions, decoder_ids)
            results = tokenizer.batch_decode(predictions)
            results = [tokenizer.convert_tokens_to_string(x) for x in results]
            results.append(src)
            results = [x.replace(' ', '') for x in results]
            results = [x.replace('[PAD]', '') for x in results]
            results = [x.replace('[CLS]', '') for x in results]
            results = [x.replace('[MASK]', '').replace('[unused3]', '').replace('[unused4]', '') for x in results]
            results = [x.split('[SEP]')[0] for x in results]
            results = [x.replace('（）', '') for x in results]
            results = [x.replace('$', '') for x in results]
            p_annos = obtain_annotation(results[1], results[0])
            if len(p_annos)==0:
                print('+++++++++++++++++++++++')
                print(results[0])
                print(results[1])
                print('+++++++++++++++++++++++')
                results[0] = results[1]
                print("skip useless modify")
            print(results[0])
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
        print(eval_ans[-1])

    result_final = {'srcs': srcs, 'prds': eval_ans, 'tars': eval_gt, 'scores': record_scores,
                    'reference': record_references}
    with open('./data/test/my_results_edit_sec_dual_%d_%d.pkl' %(config.rnn_dim, config.rnn_layer), 'wb') as f:
        pickle.dump(result_final, f)




pipieline('./data/test')
