import pickle

import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from nltk.translate.bleu_score import sentence_bleu
import nltk
from tqdm import tqdm
from datasets import load_metric
bert_model = 'hfl/chinese-bert-wwm-ext'
tokenizer = BertTokenizer.from_pretrained(bert_model)
BertScoreModel = BertModel.from_pretrained(bert_model)
BertScoreModel.cuda()
BertScoreModel.eval()
with open('../step1/data/unique_test_keys.pkl','rb') as f:
    unique_keys = pickle.load(f)
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

def cos(x, y):
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))

def get_pari_bert_f1_score(xs_pre, xs_grd):
    recall = 0.0
    precision = 0.0
    for x_pre in xs_pre:
        max_match = -1e10
        for x_grd in xs_grd:
            max_match = max(max_match, cos(x_pre, x_grd))
        precision += max_match
    precision /= len(xs_pre)

    for x_grd in xs_grd:
        max_match = -1e10
        for x_pre in xs_pre:
            max_match = max(max_match, cos(x_pre, x_grd))
        recall += max_match
    recall /= len(xs_grd)
    return recall, precision, 2*recall*precision/(recall+precision)

def get_x_list(embeddings, ids):
    xs = []
    for bid in range(len(ids)):
        temp = []
        for lid in range(len(ids[bid])):
            if ids[bid, lid] not in [101,102,0]:
                temp.append(embeddings[bid,lid])
        xs.append(temp)
    return xs

def get_hit_score(srcs, tars, pres):
    with torch.no_grad():
        recalls = []
        precisions = []
        f1s = []
        for src, tar, pre in tqdm(zip(srcs, tars, pres)):
            annotations = obtain_annotation(src, tar)
            annotations_pre = obtain_annotation(src, pre)
            if len(annotations) == 0 and len(annotations_pre) == 0:
                recall = 1
                precision = 1
            elif len(annotations_pre) == 0:
                precision = 1
                recall = 0
            elif len(annotations) == 0:
                precision = 0
                recall = 1
            else:
                annotations_ids = tokenizer(annotations,return_tensors="pt", padding=True, truncation=True)
                annotations_pre_ids = tokenizer(annotations_pre,return_tensors="pt", padding=True, truncation=True)
                annotations_embeddings = BertScoreModel(annotations_ids['input_ids'].cuda(), attention_mask=annotations_ids['attention_mask'].cuda())
                annotations_embeddings = annotations_embeddings['last_hidden_state'].cpu().numpy()
                annotations_pre_embeddings = BertScoreModel(annotations_pre_ids['input_ids'].cuda(),
                                                        attention_mask=annotations_pre_ids['attention_mask'].cuda())
                annotations_pre_embeddings = annotations_pre_embeddings['last_hidden_state'].cpu().numpy()
                annotations_xs = get_x_list(annotations_embeddings, annotations_ids['input_ids'].numpy())
                annotations_pre_xs = get_x_list(annotations_pre_embeddings, annotations_pre_ids['input_ids'].numpy())

                precision = 0
                for annotation_pre in annotations_pre_xs:
                    max_match = -1e10
                    for annotation in annotations_xs:
                        r, p, f1 = get_pari_bert_f1_score(annotation_pre, annotation)
                        max_match = max(max_match, f1)
                    precision += max_match
                precision /= len(annotations_pre_xs)

                recall = 0
                for annotation_pre in annotations_xs:
                    max_match = -1e10
                    for annotation in annotations_pre_xs:
                        r, p, f1 = get_pari_bert_f1_score(annotation_pre, annotation)
                        max_match = max(max_match, f1)
                    recall += max_match
                recall /= len(annotations_xs)
            f1 = 2 * recall * precision / (recall + precision)
            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)
        return {'anno_recall': np.mean(recalls), 'anno_precision': np.mean(precisions), 'anno_f1': np.mean(f1s)}

def get_sentence_bleu(candidate, reference):
    score = sentence_bleu(reference, candidate)
    return score
def count_bleu_score(candidate, reference):
    avg_score = 0
    for k in range(len(candidate)):
        reference_ = reference[k]
        for m in range(len(reference_)):
            reference_[m] = tokenizer.tokenize(reference_[m])
        candidate[k] = tokenizer.tokenize(candidate[k])
        try:
            avg_score += get_sentence_bleu(candidate[k], reference_)/len(candidate)
        except:
            print(candidate[k])
            print(reference[k])
    return avg_score
results = pickle.load(open('../data/test/my_results_edit_sec_limited.pkl', 'rb'))
results_temp = pickle.load(open('../data/test/my_results_bart.pkl', 'rb'))
if 'srcs' not in results:
    results['srcs'] = results_temp['srcs']
    results['tars'] = results_temp['tars']
for i in range(len(results['srcs'])-1, -1, -1):
    if len(results['srcs'][i]) <= 2 or len(results['tars'][i]) <= 2:
        del results['srcs'][i]
        del results['prds'][i]
        del results['tars'][i]
temp = []
for one in results['prds']:
    one = one.replace('(', '（')
    one = one.replace(')', '）')
    one = one.replace('。。', '。')
    temp.append(one)
results['prds'] = temp
for pr, tr in zip(results['prds'], results['tars']):
    print(pr)
    print(tr)
    print('--------------------------------------------------')
hit_score = get_hit_score(results['srcs'], results['tars'], results['prds'])
print(hit_score)
outs = [' '.join(tokenizer.tokenize(u)) for u in results['prds']]
ints = [' '.join(tokenizer.tokenize(u)) for u in results['srcs']]
refs = [[' '.join(tokenizer.tokenize(u))] for u in results['tars']]
sari = load_metric("sari")
bleu = load_metric("bleu")
meteor = load_metric('meteor')
sari_score = sari.compute(sources=ints, predictions=outs, references=refs)
print(sari_score)
meteor_score = meteor.compute(predictions=outs, references=refs)
print(meteor_score)
rouge = load_metric('rouge')
predictions = outs
references = [u[0] for u in refs]
rouge_score = rouge.compute(predictions=predictions,
                        references=references)
print(rouge_score)
outs = [tokenizer.tokenize(u) for u in results['prds']]
refs = [[tokenizer.tokenize(u)] for u in results['tars']]
bleu_score = bleu.compute(predictions=outs, references=refs)
print(bleu_score)