import pickle

import numpy as np
from transformers import BertTokenizer
from nltk.translate.bleu_score import sentence_bleu
import nltk
from datasets import load_metric
bert_model = 'fnlp/bart-base-chinese'
tokenizer = BertTokenizer.from_pretrained(bert_model)
with open('../step1/data/unique_test_keys.pkl','rb') as f:
    unique_keys = pickle.load(f)
def obtain_annotation(src, tar):
    t_s = 0
    annotations = []
    while t_s < len(tar):
        if tar[t_s] == '（':
            t_e = t_s + 1
            while t_e < len(tar) and tar[t_e] != '）' and tar[t_e] !='（':
                t_e += 1
            if t_e == len(tar):
                break
            anno_posi = tar[t_s+1:t_e]
            if len(anno_posi) > 0 and anno_posi not in src and tar[t_e] == '）':
                annotations.append(anno_posi)
            t_s = t_e
        else:
            t_s += 1
    return annotations

def get_hit_score(srcs, tars, pres):
    recalls = []
    precisions = []
    for src, tar, pre in zip(srcs, tars, pres):
        for unikey in unique_keys:
            if unikey in src:
                print(unikey)
                print('++++++++++++++')
                print(src)
                print('--------------')
                print(pre)
        annotations = obtain_annotation(src, tar)
        annotations_pre = obtain_annotation(src, pre)
        common_ones = 0
        for annotation_pre in annotations_pre:
            for annotation in annotations:
                bleu = get_sentence_bleu(tokenizer.tokenize(annotation_pre), [tokenizer.tokenize(annotation)])
                if bleu > 0.5:
                    common_ones += 1
                    break
        total = len(annotations_pre)+len(annotations)-common_ones
        score = common_ones/(len(annotations)+1e-9)
        recalls.append(score)
        score = common_ones / (len(annotations_pre) + 1e-9)
        precisions.append(score)
    return {'anno_recall': np.mean(recalls), 'anno_precision': np.mean(precisions)}

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
results = pickle.load(open('../data/test/my_results_sec_v4.pkl', 'rb'))
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
    temp.append(one)
results['prds'] = temp
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