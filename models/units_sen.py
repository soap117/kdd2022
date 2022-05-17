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
    input_ids = input_ids.numpy()
    input_ids[mask] = replace[mask]
    return torch.LongTensor(input_ids)
def check_seq(a, b):
    for x_a, x_b in zip(a, b):
        if x_a != x_b:
            return False
    return True
def find_spot(input_ids, querys_ori, tokenizer):
    positions = []
    for ori_query in querys_ori:
        flag = False
        format = '<{}>'.format(ori_query)
        format_id = tokenizer(format)['input_ids'][1:-1]
        for bid in range(input_ids.shape[0]):
            l = 0
            while input_ids[bid, l] != config.SEP and not check_seq(input_ids[bid, l:l+len(format_id)], format_id):
               l += 1
            if input_ids[bid, l] != config.SEP:
                positions.append((bid, l+len(format_id), l+len(format_id)+config.hidden_anno_len))
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
    mapping = torch.FloatTensor(mapping).to(config.device)
    scores = scores.unsqueeze(1)
    scores = scores.matmul(mapping).squeeze(1) + torch.FloatTensor(adding).to(config.device)
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
def get_retrieval_train_batch(sentences, titles, sections, bm25_title, bm25_section):
    sentences_data = []
    for sentence in tqdm(sentences):
        src_sentence = sentence['src_st']
        tar_sentence = sentence['tar_st']
        key_list = []
        for key in sentence['data']:
            region = re.search(key['origin'], src_sentence)
            if region is not None:
                region = region.regs[0]
            else:
                region = (0, 0)
            if region[0] != 0 or region[1] != 0:
                src_sentence = src_sentence[0:region[0]] + ' <{}> '.format(key['origin']) + ''.join([' [MASK] ' for x in range(config.hidden_anno_len)]) + src_sentence[region[1]:]
            region = re.search(key['origin'], tar_sentence)
            if region is not None:
                region = region.regs[0]
            else:
                region = (0, 0)
            if region[0] != 0 or region[1] != 0:
                tar_sentence = tar_sentence[0:region[0]] + ' <{}> '.format(key['origin']) + tar_sentence[region[1]:]
            data_filed = {}
            data_filed['context'] = sentence['src_st']
            if len(key['anno']) == 0:
                continue
            s = time.time()
            if len(key['rpsecs'][0]) <= 1 or len(key['key']) < 1:
                continue
            data_filed['key'] = key['key']
            data_filed['ori_key'] = key['origin']
            data_filed['anno'] = key['anno']
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
            e = time.time()
            if e-s > 5:
                print(key['key'])
            key_list.append(data_filed)
        sentences_data.append({'src_sen': src_sentence, 'tar_sen': tar_sentence, 'textid': sentence['textid'], 'key_data':key_list})
    return sentences_data


def restricted_decoding(querys_ori, srcs, tars, hidden_annotations, tokenizer, modeld):
    decoder_inputs = tokenizer(srcs, return_tensors="pt", padding=True, truncation=True)
    decoder_ids = decoder_inputs['input_ids']
    decoder_anno_positions = find_spot(decoder_ids, querys_ori, tokenizer)
    decoder_ids = decoder_ids.to(config.device)
    target_ids = tokenizer(tars, return_tensors="pt", padding=True, truncation=True)['input_ids'].to(
        config.device)
    results = []
    for bid, (src, target_id, decoder_id) in enumerate(zip(srcs, target_ids, decoder_ids)):
        decoder_anno_position = []
        h_s = 0
        h_e = 0
        for oid, one in enumerate(decoder_anno_positions):
            if one[0] == bid:
                decoder_anno_position.append((0, one[1], one[2]))
                if h_s == 0:
                    h_s = oid
                    h_e = oid + 1
                else:
                    h_e += 1
        hidden_annotation = hidden_annotations[h_s:h_e]
        final_ans = target_id[0:1]
        pointer = 1
        free_flag = False
        last_token = final_ans[-1]
        while True:
            if last_token == tokenizer.vocab['<'] or free_flag:
                outputs = modeld(input_ids=decoder_id.unsqueeze(0), decoder_input_ids=final_ans.unsqueeze(0), cut_indicator=None,
                                 anno_position=decoder_anno_position, hidden_annotation=hidden_annotation)
                logits_ = outputs.logits
                _, predictions = torch.max(logits_, dim=-1)
                next_token = predictions[0, -1]
                if not free_flag:
                    free_flag = True
                    c_count = 1
                else:
                    c_count += 1
            else:
                next_token = target_id[pointer]
                pointer += 1
            final_ans = torch.cat([final_ans, torch.LongTensor([next_token]).to(final_ans.device)], dim=0)
            if free_flag and (next_token == tokenizer.vocab['）'] or c_count > 20):
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
    def __init__(self, config, tokenizer, data_path, titles, sections, title2sections, sec2id, bm25_title, bm25_section):
        self.config = config
        sentences = read_data(data_path)
        self.title2sections = title2sections
        self.sec2id = sec2id
        self.sen_level_data = get_retrieval_train_batch(sentences, titles, sections, bm25_title, bm25_section)
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
        tar_sens = []
        cut_list = []
        c = 0
        for sen_data in train_data:
            src_sens.append(sen_data['src_sen'])
            tar_sens.append(sen_data['tar_sen'])
            for key_data in sen_data['key_data']:
                c += 1
                querys.append(key_data['key'])
                querys_ori.append(key_data['ori_key'])
                querys_context.append(sen_data['src_sen'])
                pos_title = key_data['pos_ans'][1][np.random.randint(len(key_data['pos_ans'][1]))][-1]
                pos_section = key_data['pos_ans'][0][np.random.randint(len(key_data['pos_ans'][0]))]
                sample_title_candidates = [pos_title] + key_data['neg_title_candidates']
                sample_section_candidates = [pos_section] + key_data['neg_section_candidates'] + key_data['sneg_section_candidates']
                titles.append(sample_title_candidates)
                sections.append(sample_section_candidates)
                infer_titles.append(key_data['title_candidates'])
            cut_list.append(c)
        return querys, querys_ori, querys_context, titles, sections, infer_titles, src_sens, tar_sens, cut_list

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
        tar_sens = []
        cut_list = []
        for sen_data in train_data:
            src_sens.append(sen_data['src_sen'])
            tar_sens.append(sen_data['tar_sen'])
            c = 0
            for key_data in sen_data['key_data']:
                c += 1
                querys.append(key_data['key'])
                querys_context.append(sen_data['src_sen'])
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
        return querys, querys_ori, querys_context, titles, sections, infer_titles, src_sens, tar_sens, cut_list, pos_titles, pos_sections





