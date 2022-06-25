import copy
import shlex

import torch
import numpy as np
import jieba
import time
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
smooth = SmoothingFunction()
from tqdm import tqdm
from config import config
import re
def mask_ref(input_ids, tokenizer):
    mask = np.random.choice([True, False], size=input_ids.shape, p=[0.15, 0.85])
    replace = np.random.choice(np.arange(tokenizer.vocab_size), size=input_ids.shape)
    input_ids_new = input_ids.numpy().copy()
    input_ids_new[mask] = replace[mask]
    return torch.LongTensor(input_ids_new)

def mask_actions(input_ids, tokenizer):
    mask = np.random.choice([True, False], size=input_ids.shape, p=[0.15, 0.85])
    replace = np.random.choice(np.arange(tokenizer.vocab_size), size=input_ids.shape)
    input_ids_new = input_ids.cpu().numpy().copy()
    input_ids_new[mask] = replace[mask]
    return torch.LongTensor(input_ids_new).to(input_ids.device)

def check_seq(a, b):
    for x_a, x_b in zip(a, b):
        if x_a != x_b:
            return False
    return True
def find_spot(input_ids, querys_ori, tokenizer):
    positions = []
    used_set = set()
    for ori_query in querys_ori:
        flag = False
        format = '${}$（'.format(ori_query)
        format_id = tokenizer(format)['input_ids'][1:-1]
        for bid in range(input_ids.shape[0]):
            l = 0
            while input_ids[bid, l] != config.SEP and not check_seq(input_ids[bid, l:l+len(format_id)], format_id):
               l += 1
            found_spot = (bid, l+len(format_id), l+len(format_id)+config.hidden_anno_len_rnn)
            if input_ids[bid, l] != config.SEP and found_spot not in used_set:
                positions.append(found_spot)
                used_set.add(found_spot)
                flag = True
                break
        if not flag:
            positions.append((-1,-1,-1))
    return positions

def find_spot_para(input_ids, querys_ori, tokenizer):
    positions = []
    used_set = set()
    for ori_query in querys_ori:
        flag = False
        format = '${}$（'.format(ori_query)
        format_id = tokenizer(format)['input_ids'][1:-1]
        for bid in range(input_ids.shape[0]):
            l = 0
            while input_ids[bid, l] != config.SEP and not check_seq(input_ids[bid, l:l+len(format_id)], format_id):
               l += 1
            found_spot = (bid, l+len(format_id), l+len(format_id)+config.para_hidden_len)
            if input_ids[bid, l] != config.SEP and found_spot not in used_set:
                positions.append(found_spot)
                used_set.add(found_spot)
                flag = True
                break
        if not flag:
            positions.append((-1,-1,-1))
    return positions

def batch_pointer_decode(source, pointers):
    temp = []
    source = source.cpu().numpy()
    pointers = pointers.cpu().numpy()
    for p_list, s_list in zip(pointers, source):
        temp_c = []
        for p in p_list:
            temp_c.append(s_list[p])
    temp.append(temp_c)
    return temp
def pointer_generation(source, target):
    sind = 0
    header = np.zeros(len(target))
    for wid, word in enumerate(target):
        while sind < len(source) and source[sind] != word:
            sind += 1
        if sind < len(source):
            header[wid] = sind
        else:
            sind = 0
            header[wid] = 1
    return header

def batch_pointer_generation(sources, targets):
    headers = []
    for source, target in zip(sources, targets):
        header_one = pointer_generation(source, target)
        headers.append(header_one)
    headers = np.array(headers)
    headers = torch.LongTensor(headers).to(config.device)
    return headers

def check(key, gtitles, candidata_title):
    key_cut = jieba.lcut(key)
    candidata_title = jieba.lcut(candidata_title)
    gtitles = [jieba.lcut(x) for x in gtitles]
    can_simi = sentence_bleu([candidata_title], key_cut, weights=(0.5, 0.5), smoothing_function=smooth.method1)
    grd_simi = sentence_bleu(gtitles, key_cut, weights=(0.5, 0.5), smoothing_function=smooth.method1)
    return can_simi < 0.25*grd_simi or grd_simi <= 0

def check_section(key, gsections, candidata_section, bm25, cind):
    key_cut = jieba.lcut(key)
    candidata_section = jieba.lcut(candidata_section)
    gsections = [jieba.lcut(x) for x in gsections]
    inversed_punishment = 1/np.exp(1-len(gsections[0])/len(key_cut))
    can_simi = sentence_bleu([candidata_section], key_cut, weights=(0.5, 0.5), smoothing_function=smooth.method1)*inversed_punishment
    can_simi_2 = sentence_bleu(gsections, candidata_section, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method1)
    grd_simi = sentence_bleu(gsections, key_cut, weights=(0.5, 0.5), smoothing_function=smooth.method1)*inversed_punishment
    return (can_simi < 0.25*grd_simi or grd_simi <= 0) and can_simi_2 < 0.25

def check_section_strong(gsections, candidata_section, bm25, cind):
    candidata_section = jieba.lcut(candidata_section)
    gsections = [jieba.lcut(x) for x in gsections]
    can_simi_2 = sentence_bleu(gsections, candidata_section, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method1)
    return can_simi_2 < 0.5

def neg_sample_title(key, gtitles, candidate_titles, n):
    count = 0
    try_t = 0
    rs = []
    while count < n and try_t < 100:
        ind = np.random.randint(0, len(candidate_titles))
        candidate_title = candidate_titles[ind]
        if check(key, gtitles, candidate_title):
            rs.append(candidate_title)
            count += 1
        try_t += 1
    return rs

def neg_sample_section(key, gsections, candidate_sections, n, bm25):
    count = 0
    rs = []
    try_t = 0
    while count < n and try_t < 100:
        ind = np.random.randint(0, len(candidate_sections))
        candidate_section = candidate_sections[ind]
        if len(candidate_section) < 5:
            continue
        if check_section(key, gsections, candidate_section, bm25, ind):
            rs.append(candidate_section)
            count += 1
        try_t += 1
    while count < n:
        count += 1
        rs.append("                          ")
    return rs

def neg_sample_strong_section(key, gsections, candidate_sections, n, bm25):
    count = 0
    rs = []
    try_t = 0
    while count < n and try_t < 100:
        ind = np.random.randint(0, len(candidate_sections))
        candidate_section = candidate_sections[ind]
        if len(candidate_section) < 5:
            try_t += 1
            continue
        if check_section_strong(gsections, candidate_section, bm25, ind):
            rs.append(candidate_section)
            count += 1
        try_t += 1
    while count < n:
        count += 1
        rs.append("                          ")
    return rs

def get_decoder_att_map(tokenizer, sep, ids, scores):
    spe_seq = [config.SEP]
    mapping = np.zeros([len(ids), scores.shape[1], ids.shape[1]])
    adding = np.zeros([len(ids), ids.shape[1]])
    for bindex, (bscore, bids) in enumerate(zip(scores, ids)):
        i = 0
        c_father = -1
        while i < len(bids):
            if c_father >= 0 and c_father < scores.shape[1]:
                mapping[bindex, c_father, i] = 1
            else:
                adding[bindex, i] = 1
            i += 1
            k = 0
            while k<len(spe_seq) and i<len(bids) and (spe_seq[k]==bids[i]):
                #mapping[bindex, c_father, i] = 1
                adding[bindex, i] = 1
                i += 1
                k += 1
            if k == len(spe_seq):
                c_father += 1
    mapping = torch.FloatTensor(mapping).to(ids.device)
    scores = scores.unsqueeze(1)
    scores = scores.matmul(mapping).squeeze(1) + torch.FloatTensor(adding).to(ids.device)
    return scores

def get_sec_att_map(sample_query, input_inds, infer_title_candidates, title2sections, sec2id, bm25_section):
    infer_title_candidates_pured = []
    infer_section_candidates_pured = []
    mapping_title = np.zeros([len(sample_query), 3, 4])
    for query, bid in zip(sample_query, range(len(input_inds))):
        temp = []
        temp2 = []
        temp3 = []
        for nid, cid in enumerate(input_inds[bid]):
            temp.append(infer_title_candidates[bid][cid])
            temp2 += title2sections[infer_title_candidates[bid][cid]]
            temp3 += [nid for x in title2sections[infer_title_candidates[bid][cid]]]
        temp2_id = []
        for t_sec in temp2:
            if t_sec in sec2id:
                temp2_id.append(sec2id[t_sec])
        key_cut = jieba.lcut(query)
        ls_scores = bm25_section.get_batch_scores(key_cut, temp2_id)
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
    return mapping_title, infer_title_candidates_pured, infer_section_candidates_pured
def check_useless_anno(key, src_sen, tar_sen):
    region = re.search(key, tar_sen)
    if region is not None:
        region = region.regs[0]
    else:
        return True
    region_ori = re.search(key, src_sen).regs[0]
    if tar_sen[region[1]:region[1]+2] == src_sen[region_ori[1]:region_ori[1]+2]:
        return False
    else:
        return True


def edit_distance(sent1, sent2):
    # edit from sent1 to sent2
    # Create a table to store results of subproblems
    m = len(sent1)
    n = len(sent2)
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)]
    # Fill d[][] in bottom up manner
    for i in range(m + 1):
        for j in range(n + 1):
            # If first string is empty, only option is to
            # isnert all characters of second string
            if i == 0:
                dp[i][j] = j  # Min. operations = j

            # If second string is empty, only option is to
            # remove all characters of second string
            elif j == 0:
                dp[i][j] = i  # Min. operations = i

            # If last characters are same, ignore last char
            # and recur for remaining string
            elif sent1[i - 1] == sent2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]

            # If last character are different, consider all
            # possibilities and find minimum
            else:
                edit_candidates = np.array([
                    dp[i][j - 1],  # Insert
                    dp[i - 1][j]  # Remove
                ])
                dp[i][j] = 1 + min(edit_candidates)
    return dp

def sent2edit(sent1, sent2):
    # print(sent1,sent2)
    '''
    '''
    dp = edit_distance(sent1, sent2)
    edits = []
    pos = []
    m, n = len(sent1), len(sent2)
    while m != 0 or n != 0:
        curr = dp[m][n]
        if m==0: #have to insert all here
            while n>0:
                left = dp[1][n-1]
                edits.append(sent2[n-1])
                pos.append(left)
                n-=1
        elif n==0:
            while m>0:
                top = dp[m-1][n]
                edits.append('[unused2]')
                pos.append(top)
                m -=1
        else: # we didn't reach any special cases yet
            diag = dp[m-1][n-1]
            left = dp[m][n-1]
            top = dp[m-1][n]
            if sent2[n-1] == sent1[m-1]: # keep
                edits.append('[unused1]')
                pos.append(diag)
                m -= 1
                n -= 1
            elif curr == top+1: # INSERT preferred before DEL
                edits.append('[unused2]')
                pos.append(top)  # (sent2[n-1])
                m -= 1
            else: #insert
                edits.append(sent2[n - 1])
                pos.append(left)  # (sent2[n-1])
                n -= 1
    edits = edits[::-1]
    '''
    for k in range(len(edits)-1, -1, -1):
        if edits[k] == '[unused1]':
            if edits[k-1] == '[unused1]':
                del edits[k]
        else:
            edits.append('[SEP]')
            break
    '''
    # replace the keeps at the end to stop, this helps a bit with imbalanced classes (KEEP,INS,DEL,STOP)
    ediits_ori = copy.copy(edits)


    # if edits == []: # do we learn edits if input and output are the same?
    #     edits.append('STOP') #in the case that input and output sentences are the same
    return edits, ediits_ori

def operation2sentence(operations, input_sentences):
    operations = operations.cpu().detach().numpy()
    input_sentences = input_sentences.cpu().detach().numpy()
    outputs = []
    for operation, input_sentence in zip(operations, input_sentences):
        read_index = 1
        output = [101]
        for op in operation:
            if read_index < len(input_sentence):
                if op == config.tokenizer.vocab['[unused1]']:
                    output.append(input_sentence[read_index])
                    read_index += 1
                elif op != config.tokenizer.vocab['[SEP]'] and op != config.tokenizer.vocab['[unused2]']:
                    output.append(op)
                elif op == config.tokenizer.vocab['[unused2]']:
                    read_index += 1
                    # del do nothing
                    continue
                elif op == 102:
                    break
        if read_index < len(input_sentence)-1:
            try:
                temp = list(input_sentence)[read_index:]
                output += temp
            except:
                print(output)
                print(input_sentence)
        outputs.append(output)
    return outputs

def find_UNK(ori_sentence, tokenized_sentence, tokenizer):
    new = []
    ori_sentence = ori_sentence.split()
    for i in range(len(ori_sentence)-1, -1, -1):
        if ori_sentence[i] == '':
            del ori_sentence[i]

    current_ori_index = 0
    for token in tokenized_sentence:
        if token == ori_sentence[current_ori_index]:
            new.append(token)
            current_ori_index += 1
        elif token == ori_sentence[current_ori_index][0:len(token)]:
            new.append(token)
            ori_sentence[current_ori_index] = ori_sentence[current_ori_index][len(token):]
        elif token == '[UNK]':
            sub_tokens = tokenizer.tokenize(ori_sentence[current_ori_index])
            pattern = copy.copy(ori_sentence[current_ori_index])
            for sub_token in sub_tokens:
                pattern.replace(sub_token, ' ')
            pattern = pattern.split()
            for i in range(len(pattern) - 1, -1, -1):
                if pattern[i] == '':
                    del pattern[i]
            UNK_word = pattern[0]
            new.append(UNK_word)
            ori_sentence[current_ori_index] = ori_sentence[current_ori_index][len(UNK_word):]
            if len(ori_sentence[current_ori_index]) == 0:
                current_ori_index += 1
    return new

def operation2sentence_word(operations, input_sentences, input_sequences_word, tokenizer):
    operations = operations.cpu().detach().numpy()
    input_sentences = input_sentences.cpu().detach().numpy()
    outputs = []
    out_sens = []
    for operation, input_sentence, input_sequence_word in zip(operations, input_sentences, input_sequences_word):
        out_sen = ''
        read_index = 1
        output = [101]
        input_sequence_word.append('[SEP]')
        for op in operation:
            if read_index < len(input_sentence):
                if op == config.tokenizer.vocab['[unused1]']:
                    output.append(input_sentence[read_index])
                    out_sen += input_sequence_word[read_index-1]
                    out_sen += ' '
                    read_index += 1
                elif op != config.tokenizer.vocab['[SEP]'] and op != config.tokenizer.vocab['[unused2]']:
                    output.append(op)
                    out_sen += tokenizer.ids_to_tokens[op]
                    out_sen += ' '
                elif op == config.tokenizer.vocab['[unused2]']:
                    read_index += 1
                    # del do nothing
                    continue
                elif op == 102:
                    break
        if read_index < len(input_sentence)-1:
            try:
                temp = list(input_sentence)[read_index:]
                output += temp
                for word in input_sequence_word[read_index-1:]:
                    out_sen += word
                    out_sen += ' '
            except:
                print(output)
                print(input_sentence)
        outputs.append(output)
        out_sens.append(out_sen)
    return outputs, out_sens

def obtain_annotation(tar , t_s):
    t_e = t_s + 1
    count = 1
    while t_e < len(tar) and count > 0:
        if tar[t_e] == '）':
            count -= 1
        if tar[t_e] == '（':
            count += 1
        t_e += 1
    annotations = tar[t_s+1:t_e-1]

    return annotations
def is_in_annotation(pos, src):
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

def obtain_annotations(tar):
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
            anno_posi = tar[t_s:t_e]
            annotations.append(anno_posi)
            t_s = t_e
        else:
            t_s += 1
    return annotations

def find_context_sentence(r, para):
    l = r-1
    r = r+1
    while l > 0 and para[l] != '。':
        l -= 1
    while r < len(para) and para[r] != '。':
        r += 1
    context = para[l+1:r]
    return context

def get_retrieval_train_batch(sentences, titles, sections, bm25_title, bm25_section, wo_re=False):
    sentences_data = []
    for sentence in tqdm(sentences):
        src_sentence = sentence['src_st']
        src_sentence_ori = copy.copy(src_sentence)
        tar_sentence = sentence['tar_st']
        key_list = []
        temp_keys = sentence['data']
        temp_keys = sorted(temp_keys, key=lambda x: len(x['origin']), reverse=True)
        used = []
        if wo_re:
            temp_keys = []
        for key in temp_keys:
            flag = False
            for key_used in used:
                if key['origin'] in key_used:
                    flag = True
                    break
            if flag:
                continue
            region = re.search(key['origin'], src_sentence)
            if region is not None:
                region = region.regs[0]
            else:
                region = (0, 0)
            if region[0] != 0 or region[1] != 0:
                src_sentence = src_sentence[0:region[0]] + '${}$'.format(key['origin']) + '（' + ''.join([' [unused3] ']+[' [MASK] ' for x in range(config.hidden_anno_len_rnn-2)] + [' [unused4] ']) + '）' + src_sentence[region[1]:]
            regions = [x for x in re.finditer(key['origin'], tar_sentence)]
            region = None
            for one in regions:
                if is_in_annotation(one.regs[0][0], tar_sentence):
                    continue
                region = one.regs[0]
                break
            if region is None and len(regions) == 1:
                region = regions[0].regs[0]
            if region is not None:
                region = region
            else:
                region = (0, 0)
            if region[0] != 0 or region[1] != 0:
                if region[1] < len(tar_sentence) and tar_sentence[region[1]] != '（':
                    tar_sentence = tar_sentence[0:region[0]] + '${}$（）'.format(key['origin']) + tar_sentence[region[1]:]
                elif region[1] < len(tar_sentence) and tar_sentence[region[1]] == '（':
                    annotation = obtain_annotation(tar_sentence, region[1])
                    if annotation in src_sentence:
                        tar_sentence = tar_sentence[0:region[0]] + '${}$（）'.format(key['origin']) + tar_sentence[region[1]:]
                    else:
                        tar_sentence = tar_sentence[0:region[0]] + '${}$'.format(key['origin']) + tar_sentence[region[1]:]
                else:
                    tar_sentence = tar_sentence[0:region[0]] + '${}$'.format(key['origin']) + tar_sentence[region[1]:]

            data_filed = {}
            data_filed['context'] = sentence['src_st']
            region_ori = re.search(key['origin'], src_sentence_ori)
            if region_ori is not None:
                region_ori = region_ori.regs[0]
                context_key = find_context_sentence(region_ori[0], src_sentence_ori)
            else:
                region_ori = (0, 0)
                context_key = '        '
            if len(context_key) > 100:
                context_key = context_key[0:100]
            if len(key['anno']) == 0:
                continue
            s = time.time()
            if len(key['rpsecs'][0]) <= 1 or len(key['key']) < 1:
                continue
            data_filed['key'] = key['key']
            data_filed['ori_key'] = key['origin']
            data_filed['anno'] = key['anno']
            data_filed['context'] = context_key

            key_cut = jieba.lcut(key['key'])
            infer_titles = bm25_title.get_top_n(key_cut, titles, config.infer_title_range)
            data_filed['title_candidates'] = infer_titles
            for x in key['rpsecs']:
                if len(x) == 0:
                    x.append('')
            neg_titles = neg_sample_title(key['key'], [x[-1] for x in key['rpsecs']], titles, config.neg_num)
            neg_sections = neg_sample_section(key['key'], key['rsecs'], sections, config.neg_num, bm25_section)
            temp_strong_neg_sections = []
            for _ in key['rpsecs']:
                temp_strong_neg_sections += _
            neg_sections_strong = neg_sample_strong_section(key['key'], key['rsecs'], temp_strong_neg_sections, config.neg_strong_num, bm25_section)
            if len(neg_sections_strong) <= 0:
                neg_sections_strong.append('                                                            ')
            #pos_section = key['rsecs'][np.random.randint(len(key['rsecs']))]
            #pos_title = key['rpsecs'][np.random.randint(len(key['rpsecs']))][-1]
            data_filed['pos_ans'] = (key['rsecs'], key['rpsecs'])
            data_filed['neg_title_candidates'] = neg_titles
            data_filed['neg_section_candidates'] = neg_sections
            data_filed['sneg_section_candidates'] = neg_sections_strong
            used.append(key['origin'])
            e = time.time()
            if e-s > 5:
                print(key['key'])
            key_list.append(data_filed)
        src_tokens = config.tokenizer.tokenize(src_sentence)
        src_tokens_ori = config.tokenizer.tokenize(src_sentence_ori)
        tar_tokens = config.tokenizer.tokenize(tar_sentence)
        # check
        '''
        tar_ids = config.tokenizer.convert_tokens_to_ids(tar_tokens)
        tar_ids_2 = config.tokenizer(tar_sentence)['input_ids'][1:-1]
        if tar_ids != tar_ids_2:
            print('here')
         '''
        edit_tokens, edit_tokens_ori = sent2edit(src_tokens, tar_tokens)
        tar_sentence_clean = tar_sentence.replace('$', '')
        tar_annos = obtain_annotations(tar_sentence_clean)
        for tar_anno in tar_annos:
            tar_sentence_clean = tar_sentence_clean.replace(tar_anno, '')
        src_sentence_clean = src_sentence_ori.replace('$', '')
        src_annos = obtain_annotations(src_sentence_clean)
        for src_anno in src_annos:
            src_sentence_clean = src_sentence_clean.replace(src_anno, '')
        if len(tar_sentence_clean)>2*len(src_sentence_clean) or len(src_sentence_clean)>2*len(tar_sentence_clean):
            print('Not a good match')
            print(src_sentence_clean)
            print(tar_sentence_clean)
            continue
        sentences_data.append({'src_sen': src_sentence, 'src_sen_ori': src_sentence_ori,
                               'tar_sen': tar_sentence, 'textid': sentence['textid'], 'key_data':key_list, 'edit_sen':edit_tokens})
    return sentences_data

def get_retrieval_train_batch_word(sentences, titles, sections, bm25_title, bm25_section, wo_re=False, is_pure=False):
    if is_pure:
        sentences_data = []
        for sentence in tqdm(sentences):
            src_sentence = sentence['src_st']
            src_sentence_ori = copy.copy(src_sentence)
            tar_sentence = sentence['tar_st']
            key_list = []
            temp_keys = sentence['data']
            temp_keys = sorted(temp_keys, key=lambda x: len(x['origin']), reverse=True)
            used = []
            if wo_re:
                temp_keys = []
            for key in temp_keys:
                flag = False
                for key_used in used:
                    if key['origin'] in key_used:
                        flag = True
                        break
                if flag:
                    continue
                regions = [x for x in re.finditer(key['origin'], tar_sentence)]
                region = None
                for one in regions:
                    if is_in_annotation(one.regs[0][0], tar_sentence):
                        continue
                    region = one.regs[0]
                    break
                if region is None and len(regions) == 1:
                    region = regions[0].regs[0]
                if region is not None:
                    region = region
                else:
                    region = (0, 0)
                if region[0] != 0 or region[1] != 0:
                    if region[1] < len(tar_sentence) and tar_sentence[region[1]] == '（':
                        annotation = obtain_annotation(tar_sentence, region[1])
                        if annotation not in src_sentence:
                            tar_sentence = tar_sentence[0:region[0]] + '${}$'.format(key['origin']) + tar_sentence[
                                                                                                      region[1]:]
                            region = re.search(key['origin'], src_sentence)
                            if region is not None:
                                region = region.regs[0]
                            else:
                                region = (0, 0)
                            if region[0] != 0 or region[1] != 0:
                                src_sentence = src_sentence[0:region[0]] + '${}$'.format(key['origin']) + '（' + ''.join(
                                    [' [unused3] '] + [' [MASK] ' for x in range(config.para_hidden_len - 2)] + [
                                        ' [unused4] ']) + '）' + src_sentence[region[1]:]

                data_filed = {}
                data_filed['context'] = sentence['src_st']
                region_ori = re.search(key['origin'], src_sentence_ori)
                if region_ori is not None:
                    region_ori = region_ori.regs[0]
                    context_key = find_context_sentence(region_ori[0], src_sentence_ori)
                else:
                    region_ori = (0, 0)
                    context_key = '        '
                if len(context_key) > 100:
                    context_key = context_key[0:100]
                if len(key['anno']) == 0:
                    continue
                s = time.time()
                if len(key['rpsecs'][0]) <= 1 or len(key['key']) < 1:
                    continue
                data_filed['key'] = key['key']
                data_filed['ori_key'] = key['origin']
                data_filed['anno'] = key['anno']
                data_filed['context'] = context_key

                key_cut = jieba.lcut(key['key'])
                infer_titles = bm25_title.get_top_n(key_cut, titles, config.infer_title_range)
                data_filed['title_candidates'] = infer_titles
                for x in key['rpsecs']:
                    if len(x) == 0:
                        x.append('')
                neg_titles = neg_sample_title(key['key'], [x[-1] for x in key['rpsecs']], titles, config.neg_num)
                neg_sections = neg_sample_section(key['key'], key['rsecs'], sections, config.neg_num, bm25_section)
                temp_strong_neg_sections = []
                for _ in key['rpsecs']:
                    temp_strong_neg_sections += _
                neg_sections_strong = neg_sample_strong_section(key['key'], key['rsecs'], temp_strong_neg_sections,
                                                                config.neg_strong_num, bm25_section)
                if len(neg_sections_strong) <= 0:
                    neg_sections_strong.append('                                                            ')
                # pos_section = key['rsecs'][np.random.randint(len(key['rsecs']))]
                # pos_title = key['rpsecs'][np.random.randint(len(key['rpsecs']))][-1]
                data_filed['pos_ans'] = (key['rsecs'], key['rpsecs'])
                data_filed['neg_title_candidates'] = neg_titles
                data_filed['neg_section_candidates'] = neg_sections
                data_filed['sneg_section_candidates'] = neg_sections_strong
                used.append(key['origin'])
                e = time.time()
                if e - s > 5:
                    print(key['key'])
                key_list.append(data_filed)
            src_tokens = config.tokenizer_editplus.tokenize(config.pre_cut(src_sentence))
            tar_tokens = config.tokenizer_editplus.tokenize(config.pre_cut(tar_sentence))
            # check
            '''
            tar_ids = config.tokenizer.convert_tokens_to_ids(tar_tokens)
            tar_ids_2 = config.tokenizer(tar_sentence)['input_ids'][1:-1]
            if tar_ids != tar_ids_2:
                print('here')
             '''
            edit_tokens, edit_tokens_ori = sent2edit(src_tokens, tar_tokens)
            tar_sentence_clean = tar_sentence.replace('$', '')
            tar_annos = obtain_annotations(tar_sentence_clean)
            for tar_anno in tar_annos:
                tar_sentence_clean = tar_sentence_clean.replace(tar_anno, '')
            src_sentence_clean = src_sentence_ori.replace('$', '')
            src_annos = obtain_annotations(src_sentence_clean)
            for src_anno in src_annos:
                src_sentence_clean = src_sentence_clean.replace(src_anno, '')
            if len(tar_sentence_clean) > 2 * len(src_sentence_clean) or len(src_sentence_clean) > 2 * len(
                    tar_sentence_clean):
                print('Not a good match')
                print(src_sentence_clean)
                print(tar_sentence_clean)
                continue
            sentences_data.append({'src_sen': config.pre_cut(src_sentence.lower()), 'src_sen_ori': src_sentence_ori,
                                   'tar_sen': config.pre_cut(tar_sentence.lower()), 'textid': sentence['textid'], 'key_data': key_list,
                                   'edit_sen': edit_tokens})
    else:
        sentences_data = []
        for sentence in tqdm(sentences):
            src_sentence = sentence['src_st']
            src_sentence_ori = copy.copy(src_sentence)
            tar_sentence = sentence['tar_st']
            key_list = []
            temp_keys = sentence['data']
            temp_keys = sorted(temp_keys, key=lambda x: len(x['origin']), reverse=True)
            used = []
            if wo_re:
                temp_keys = []
            for key in temp_keys:
                flag = False
                for key_used in used:
                    if key['origin'] in key_used:
                        flag = True
                        break
                if flag:
                    continue
                region = re.search(key['origin'], src_sentence)
                if region is not None:
                    region = region.regs[0]
                else:
                    region = (0, 0)
                if region[0] != 0 or region[1] != 0:
                    src_sentence = src_sentence[0:region[0]] + '${}$'.format(key['origin']) + '（' + ''.join([' [unused3] ']+[' [MASK] ' for x in range(config.hidden_anno_len_rnn-2)] + [' [unused4] ']) + '）' + src_sentence[region[1]:]
                regions = [x for x in re.finditer(key['origin'], tar_sentence)]
                region = None
                for one in regions:
                    if is_in_annotation(one.regs[0][0], tar_sentence):
                        continue
                    region = one.regs[0]
                    break
                if region is None and len(regions) == 1:
                    region = regions[0].regs[0]
                if region is not None:
                    region = region
                else:
                    region = (0, 0)
                if region[0] != 0 or region[1] != 0:
                    if region[1] < len(tar_sentence) and tar_sentence[region[1]] != '（':
                        tar_sentence = tar_sentence[0:region[0]] + '${}$（）'.format(key['origin']) + tar_sentence[region[1]:]
                    elif region[1] < len(tar_sentence) and tar_sentence[region[1]] == '（':
                        annotation = obtain_annotation(tar_sentence, region[1])
                        if annotation in src_sentence:
                            tar_sentence = tar_sentence[0:region[0]] + '${}$（）'.format(key['origin']) + tar_sentence[region[1]:]
                        else:
                            tar_sentence = tar_sentence[0:region[0]] + '${}$'.format(key['origin']) + tar_sentence[region[1]:]
                    else:
                        tar_sentence = tar_sentence[0:region[0]] + '${}$'.format(key['origin']) + tar_sentence[region[1]:]

                data_filed = {}
                data_filed['context'] = sentence['src_st']
                region_ori = re.search(key['origin'], src_sentence_ori)
                if region_ori is not None:
                    region_ori = region_ori.regs[0]
                    context_key = find_context_sentence(region_ori[0], src_sentence_ori)
                else:
                    region_ori = (0, 0)
                    context_key = '        '
                if len(context_key) > 100:
                    context_key = context_key[0:100]
                if len(key['anno']) == 0:
                    continue
                s = time.time()
                if len(key['rpsecs'][0]) <= 1 or len(key['key']) < 1:
                    continue
                data_filed['key'] = key['key']
                data_filed['ori_key'] = key['origin']
                data_filed['anno'] = key['anno']
                data_filed['context'] = context_key

                key_cut = jieba.lcut(key['key'])
                infer_titles = bm25_title.get_top_n(key_cut, titles, config.infer_title_range)
                data_filed['title_candidates'] = infer_titles
                for x in key['rpsecs']:
                    if len(x) == 0:
                        x.append('')
                neg_titles = neg_sample_title(key['key'], [x[-1] for x in key['rpsecs']], titles, config.neg_num)
                neg_sections = neg_sample_section(key['key'], key['rsecs'], sections, config.neg_num, bm25_section)
                temp_strong_neg_sections = []
                for _ in key['rpsecs']:
                    temp_strong_neg_sections += _
                neg_sections_strong = neg_sample_strong_section(key['key'], key['rsecs'], temp_strong_neg_sections, config.neg_strong_num, bm25_section)
                if len(neg_sections_strong) <= 0:
                    neg_sections_strong.append('                                                            ')
                #pos_section = key['rsecs'][np.random.randint(len(key['rsecs']))]
                #pos_title = key['rpsecs'][np.random.randint(len(key['rpsecs']))][-1]
                data_filed['pos_ans'] = (key['rsecs'], key['rpsecs'])
                data_filed['neg_title_candidates'] = neg_titles
                data_filed['neg_section_candidates'] = neg_sections
                data_filed['sneg_section_candidates'] = neg_sections_strong
                used.append(key['origin'])
                e = time.time()
                if e-s > 5:
                    print(key['key'])
                key_list.append(data_filed)
            src_tokens = config.tokenizer_editplus.tokenize(config.pre_cut(src_sentence))
            tar_tokens = config.tokenizer_editplus.tokenize(config.pre_cut(tar_sentence))
            # check
            '''
            tar_ids = config.tokenizer.convert_tokens_to_ids(tar_tokens)
            tar_ids_2 = config.tokenizer(tar_sentence)['input_ids'][1:-1]
            if tar_ids != tar_ids_2:
                print('here')
             '''
            edit_tokens, edit_tokens_ori = sent2edit(src_tokens, tar_tokens)
            tar_sentence_clean = tar_sentence.replace('$', '')
            tar_annos = obtain_annotations(tar_sentence_clean)
            for tar_anno in tar_annos:
                tar_sentence_clean = tar_sentence_clean.replace(tar_anno, '')
            src_sentence_clean = src_sentence_ori.replace('$', '')
            src_annos = obtain_annotations(src_sentence_clean)
            for src_anno in src_annos:
                src_sentence_clean = src_sentence_clean.replace(src_anno, '')
            if len(tar_sentence_clean)>2*len(src_sentence_clean) or len(src_sentence_clean)>2*len(tar_sentence_clean):
                print('Not a good match')
                print(src_sentence_clean)
                print(tar_sentence_clean)
                continue
            sentences_data.append({'src_sen': config.pre_cut(src_sentence.lower()), 'src_sen_ori': src_sentence_ori,
                                   'tar_sen': config.pre_cut(tar_sentence.lower()), 'textid': sentence['textid'], 'key_data':key_list, 'edit_sen':edit_tokens})
    return sentences_data

def get_retrieval_train_batch_pure(sentences, titles, sections, bm25_title, bm25_section, wo_re=False):
    sentences_data = []
    for sentence in tqdm(sentences):
        src_sentence = sentence['src_st']
        src_sentence_ori = copy.copy(src_sentence)
        tar_sentence = sentence['tar_st']
        key_list = []
        temp_keys = sentence['data']
        temp_keys = sorted(temp_keys, key=lambda x: len(x['origin']), reverse=True)
        used = []
        if wo_re:
            temp_keys = []
        for key in temp_keys:
            flag = False
            for key_used in used:
                if key['origin'] in key_used:
                    flag = True
                    break
            if flag:
                continue
            regions = [x for x in re.finditer(key['origin'], tar_sentence)]
            region = None
            for one in regions:
                if is_in_annotation(one.regs[0][0], tar_sentence):
                    continue
                region = one.regs[0]
                break
            if region is None and len(regions) == 1:
                region = regions[0].regs[0]
            if region is not None:
                region = region
            else:
                region = (0, 0)
            if region[0] != 0 or region[1] != 0:
                if region[1] < len(tar_sentence) and tar_sentence[region[1]] == '（':
                    annotation = obtain_annotation(tar_sentence, region[1])
                    if annotation not in src_sentence:
                        tar_sentence = tar_sentence[0:region[0]] + '${}$'.format(key['origin']) + tar_sentence[region[1]:]
                        region = re.search(key['origin'], src_sentence)
                        if region is not None:
                            region = region.regs[0]
                        else:
                            region = (0, 0)
                        if region[0] != 0 or region[1] != 0:
                            src_sentence = src_sentence[0:region[0]] + '${}$'.format(key['origin']) + '（' + ''.join(
                                [' [unused3] '] + [' [MASK] ' for x in range(config.para_hidden_len - 2)] + [
                                    ' [unused4] ']) + '）' + src_sentence[region[1]:]

            data_filed = {}
            data_filed['context'] = sentence['src_st']
            region_ori = re.search(key['origin'], src_sentence_ori)
            if region_ori is not None:
                region_ori = region_ori.regs[0]
                context_key = find_context_sentence(region_ori[0], src_sentence_ori)
            else:
                region_ori = (0, 0)
                context_key = '        '
            if len(context_key) > 100:
                context_key = context_key[0:100]
            if len(key['anno']) == 0:
                continue
            s = time.time()
            if len(key['rpsecs'][0]) <= 1 or len(key['key']) < 1:
                continue
            data_filed['key'] = key['key']
            data_filed['ori_key'] = key['origin']
            data_filed['anno'] = key['anno']
            data_filed['context'] = context_key

            key_cut = jieba.lcut(key['key'])
            infer_titles = bm25_title.get_top_n(key_cut, titles, config.infer_title_range)
            data_filed['title_candidates'] = infer_titles
            for x in key['rpsecs']:
                if len(x) == 0:
                    x.append('')
            neg_titles = neg_sample_title(key['key'], [x[-1] for x in key['rpsecs']], titles, config.neg_num)
            neg_sections = neg_sample_section(key['key'], key['rsecs'], sections, config.neg_num, bm25_section)
            temp_strong_neg_sections = []
            for _ in key['rpsecs']:
                temp_strong_neg_sections += _
            neg_sections_strong = neg_sample_strong_section(key['key'], key['rsecs'], temp_strong_neg_sections, config.neg_strong_num, bm25_section)
            if len(neg_sections_strong) <= 0:
                neg_sections_strong.append('                                                            ')
            #pos_section = key['rsecs'][np.random.randint(len(key['rsecs']))]
            #pos_title = key['rpsecs'][np.random.randint(len(key['rpsecs']))][-1]
            data_filed['pos_ans'] = (key['rsecs'], key['rpsecs'])
            data_filed['neg_title_candidates'] = neg_titles
            data_filed['neg_section_candidates'] = neg_sections
            data_filed['sneg_section_candidates'] = neg_sections_strong
            used.append(key['origin'])
            e = time.time()
            if e-s > 5:
                print(key['key'])
            key_list.append(data_filed)
        src_tokens = config.tokenizer.tokenize(src_sentence)
        src_tokens_ori = config.tokenizer.tokenize(src_sentence_ori)
        tar_tokens = config.tokenizer.tokenize(tar_sentence)
        # check
        '''
        tar_ids = config.tokenizer.convert_tokens_to_ids(tar_tokens)
        tar_ids_2 = config.tokenizer(tar_sentence)['input_ids'][1:-1]
        if tar_ids != tar_ids_2:
            print('here')
         '''
        edit_tokens, edit_tokens_ori = sent2edit(src_tokens, tar_tokens)
        tar_sentence_clean = tar_sentence.replace('$', '')
        tar_annos = obtain_annotations(tar_sentence_clean)
        for tar_anno in tar_annos:
            tar_sentence_clean = tar_sentence_clean.replace(tar_anno, '')
        src_sentence_clean = src_sentence_ori.replace('$', '')
        src_annos = obtain_annotations(src_sentence_clean)
        for src_anno in src_annos:
            src_sentence_clean = src_sentence_clean.replace(src_anno, '')
        if len(tar_sentence_clean)>2*len(src_sentence_clean) or len(src_sentence_clean)>2*len(tar_sentence_clean):
            print('Not a good match')
            print(src_sentence_clean)
            print(tar_sentence_clean)
            continue
        sentences_data.append({'src_sen': src_sentence, 'src_sen_ori': src_sentence_ori,
                               'tar_sen': tar_sentence, 'textid': sentence['textid'], 'key_data':key_list, 'edit_sen':edit_tokens})
    return sentences_data



def restricted_decoding(querys_ori, srcs, tars, hidden_annotations, tokenizer, modeld, is_free=False):
    decoder_inputs = tokenizer(srcs, return_tensors="pt", padding=True, truncation=True)
    decoder_ids = decoder_inputs['input_ids']
    decoder_anno_positions = find_spot(decoder_ids, querys_ori, tokenizer)
    decoder_ids = decoder_ids.to(config.device)
    target_ids = tokenizer(tars, return_tensors="pt", padding=True, truncation=True)['input_ids'].to(
        config.device)
    results = []
    hidden_annotations_en = modeld.hidden_annotation_alignment(hidden_annotations)
    encoder_outputs = modeld.model.encoder(
        input_ids=decoder_ids,
        anno_position=decoder_anno_positions,
        hidden_annotations=hidden_annotations_en,
    )
    for bid, (src, target_id, decoder_id) in enumerate(zip(srcs, target_ids, decoder_ids)):
        final_ans = target_id[0:1]
        pointer = 1
        free_flag = False
        last_token = final_ans[-1]
        while True:
            if last_token == tokenizer.vocab['$'] or free_flag or is_free:
                decoder_outputs = modeld.model.decoder(
                    input_ids=final_ans.unsqueeze(0),
                    encoder_hidden_states=encoder_outputs[0][bid].unsqueeze(0),
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                    use_cache=True
                )
                lm_logits = modeld.lm_head(decoder_outputs.last_hidden_state) + modeld.final_logits_bias
                logits_ = lm_logits
                _, predictions = torch.max(logits_, dim=-1)
                next_token = predictions[0, -1]
                if not free_flag:
                    free_flag = True
                    c_count = 1
                    try:
                        while target_id[pointer] != tokenizer.vocab['）'] and not is_free:
                            pointer += 1
                    except:
                        print(tokenizer.decode(target_id))
                        exit(-1)
                    pointer += 1
                    cons_pointer = pointer
                    cons_count = 0
                else:
                    c_count += 1
            else:
                next_token = target_id[pointer]
                pointer += 1
            final_ans = torch.cat([final_ans, torch.LongTensor([next_token]).to(final_ans.device)], dim=0)
            if free_flag and (next_token == tokenizer.vocab['）'] or c_count > 30) and not is_free:
                next_token = target_id[pointer]
                pointer += 1
                final_ans = torch.cat([final_ans, torch.LongTensor([next_token]).to(final_ans.device)], dim=0)
                if c_count > 30:
                    final_ans = torch.cat([final_ans, torch.LongTensor([tokenizer.vocab['）']]).to(final_ans.device)], dim=0)
                free_flag = False
            if free_flag and next_token == target_id[cons_pointer] and not is_free:
                cons_count += 1
                cons_pointer += 1
                if cons_count == 3:
                    pointer = cons_pointer
                    free_flag = False
            last_token = final_ans[-1]
            if last_token == tokenizer.vocab['[SEP]'] or len(final_ans) >= config.maxium_sec or pointer >= len(target_id):
                break
        result = tokenizer.decode(final_ans)
        result = result.replace(' ', '')
        result = result.replace('[PAD]', '')
        result = result.replace('[CLS]', '')
        result = result.split('[SEP]')[0]
        results.append(result)
    return results, target_ids


from torch.utils.data import Dataset
import pickle, re
def read_clean_data(path):
    sample_data = pickle.load(open(path, 'rb'))
    titles = []
    sections = []
    title2sections = {}
    urls = set()
    sec2id = {}
    for one_sen in sample_data:
        for one in one_sen['data']:
            if len(one['urls']) > 0:
                for tid, (title, url, tref) in enumerate(zip(one['rpsecs'], one['urls'], one['rsecs'])):
                    if len(title) > 0:
                        web_title = title[-1]
                        web_title = re.sub('_.+', '', web_title)
                        web_title = re.sub(' -.+', '', web_title)
                        one['rpsecs'][tid][-1] = web_title
                        sections += title[0:-1]
                        titles.append(web_title)
                        if web_title in title2sections and url not in urls:
                            title2sections[web_title] += title[0:-1]
                            if tref not in title2sections[web_title]:
                                title2sections[web_title].append(tref)
                            urls.add(url)
                        elif web_title not in title2sections:
                            title2sections[web_title] = title[0:-1]
                            if tref not in title2sections[web_title]:
                                title2sections[web_title].append(tref)
                            urls.add(url)

    titles = list(set(titles))
    sections = list(set(sections))
    for k in range(len(sections)-1, -1, -1):
        if len(sections[k]) < 5:
            del sections[k]
    for tid, temp in enumerate(sections):
        sec2id[temp] = tid
    return titles, sections, title2sections, sec2id

def read_data(path):
    sample_data = pickle.load(open(path, 'rb'))
    sentences = []
    for one_sen in sample_data:
        for one in one_sen['data']:
            if len(one['urls']) > 0:
                for tid, (title, url) in enumerate(zip(one['rpsecs'], one['urls'])):
                    if len(title) > 0:
                        web_title = title[-1]
                        web_title = re.sub('_.+', '', web_title)
                        web_title = re.sub(' -.+', '', web_title)
                        one['rpsecs'][tid][-1] = web_title
        sentences.append(one_sen)
    return sentences


class MyData(Dataset):
    def __init__(self, config, tokenizer, data_path, titles, sections, title2sections, sec2id, bm25_title, bm25_section, is_pure=False, wo_re=False, word=False):
        self.config = config
        sentences = read_data(data_path)
        self.title2sections = title2sections
        self.sec2id = sec2id
        if word:
            self.sen_level_data = get_retrieval_train_batch_word(sentences, titles, sections, bm25_title, bm25_section,
                                                            wo_re, is_pure=is_pure)
        elif is_pure:
            self.sen_level_data = get_retrieval_train_batch_pure(sentences, titles, sections, bm25_title, bm25_section, wo_re)
        else:
            self.sen_level_data = get_retrieval_train_batch(sentences, titles, sections, bm25_title, bm25_section, wo_re)
        self.bm25_title = bm25_title
        self.bm25_section = bm25_section
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sen_level_data)

    def __getitem__(self, item):
        return self.sen_level_data[item]

    def collate_fn(self, train_data):
        querys = []
        querys_ori = []
        querys_context = []
        titles = []
        sections = []
        infer_titles = []
        src_sens = []
        src_sens_ori = []
        tar_sens = []
        edit_sens = []
        cut_list = []
        c = 0
        for sen_data in train_data:
            src_sens.append(sen_data['src_sen'])
            src_sens_ori.append(sen_data['src_sen_ori'])
            tar_sens.append(sen_data['tar_sen'])
            edit_sens.append(sen_data['edit_sen'])
            index_key = np.arange(0, len(sen_data['key_data']))
            index_key_filtered = np.sort(np.random.choice(index_key, min(len(index_key), config.max_query), replace=False))
            for key_index in index_key_filtered:
                key_data = sen_data['key_data'][key_index]
                c += 1
                querys.append(key_data['key'])
                querys_ori.append(key_data['ori_key'])
                querys_context.append(key_data['context'])
                pos_title = key_data['pos_ans'][1][np.random.randint(len(key_data['pos_ans'][1]))][-1]
                pos_section = key_data['pos_ans'][0][np.random.randint(len(key_data['pos_ans'][0]))]
                sample_title_candidates = [pos_title] + key_data['neg_title_candidates']
                sample_section_candidates = [pos_section] + key_data['neg_section_candidates'] + key_data['sneg_section_candidates']
                titles.append(sample_title_candidates)
                sections.append(sample_section_candidates)
                infer_titles.append(key_data['title_candidates'])
            cut_list.append(c)
        return querys, querys_ori, querys_context, titles, sections, infer_titles, src_sens, src_sens_ori, tar_sens, cut_list, edit_sens

    def collate_fn_test(self, train_data):
        pos_titles = []
        pos_sections = []
        querys = []
        querys_ori = []
        querys_context = []
        titles = []
        sections = []
        infer_titles = []
        src_sens = []
        src_sens_ori = []
        tar_sens = []
        cut_list = []
        edit_sens = []
        for sen_data in train_data:
            src_sens.append(sen_data['src_sen'])
            src_sens_ori.append(sen_data['src_sen_ori'])
            tar_sens.append(sen_data['tar_sen'])
            edit_sens.append(sen_data['edit_sen'])
            c = 0
            for key_data in sen_data['key_data']:
                c += 1
                querys.append(key_data['key'])
                querys_context.append(sen_data['src_sen_ori'])
                querys_ori.append(key_data['ori_key'])
                pos_title = key_data['pos_ans'][1][np.random.randint(len(key_data['pos_ans'][1]))][-1]
                pos_section = key_data['pos_ans'][0][np.random.randint(len(key_data['pos_ans'][0]))]
                sample_title_candidates = [pos_title] + key_data['neg_title_candidates']
                sample_section_candidates = [pos_section] + key_data['neg_section_candidates'] + key_data[
                    'sneg_section_candidates']
                titles.append(sample_title_candidates)
                sections.append(sample_section_candidates)
                infer_titles.append(key_data['title_candidates'])

                pos_title_all = [x[-1] for x in key_data['pos_ans'][1]]
                pos_section_all = key_data['pos_ans'][0]
                pos_titles.append(pos_title_all)
                pos_sections.append(pos_section_all)
            cut_list.append(c)
        return querys, querys_ori, querys_context, titles, sections, infer_titles, src_sens, src_sens_ori, tar_sens, cut_list, pos_titles, pos_sections, edit_sens





