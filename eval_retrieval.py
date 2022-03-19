import torch
from config import config
from models.units import MyData, get_decoder_att_map
from models.bert_tokenizer import BertTokenizer
from torch.utils.data import DataLoader
from models.retrieval import TitleEncoder, PageRanker, SecEncoder, SectionRanker
from models.modeling_gpt2_att import GPT2LMHeadModel
from tqdm import tqdm
from transformers import AdamW
import numpy as np
import jieba
from models.units import read_clean_data
from rank_bm25 import BM25Okapi

def build(config):
    save_data = torch.load('./results/best_model.bin', map_location=torch.device('cuda:0'))
    tokenizer = BertTokenizer(vocab_file='./GPT2Chinese/vocab.txt', do_lower_case=False, never_split=['[SEP]'])
    titles, sections, title2sections, sec2id = read_clean_data('data/mydata_new_baidu_.pkl')
    corpus = sections
    tokenized_corpus = [jieba.lcut(doc) for doc in corpus]
    bm25_section = BM25Okapi(tokenized_corpus)

    corpus = titles
    tokenized_corpus = [jieba.lcut(doc) for doc in corpus]
    bm25_title = BM25Okapi(tokenized_corpus)
    train_dataset = None
    valid_dataset = MyData(config, tokenizer, 'data/valid.pkl', titles, sections, title2sections, sec2id, bm25_title,
                           bm25_section)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=config.batch_size
                                  , collate_fn=valid_dataset.collate_fn)
    test_dataset = MyData(config, tokenizer, 'data/test.pkl', titles, sections, title2sections, sec2id, bm25_title,
                           bm25_section)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=config.batch_size
                                  , collate_fn=test_dataset.collate_fn)

    modelp = save_data['modelp']
    models = save_data['models']
    model = save_data['model']
    optimizer_p = AdamW(modelp.parameters(), lr=config.lr)
    optimizer_s = AdamW(models.parameters(), lr=config.lr)
    optimizer_decoder = None
    loss_func = torch.nn.CrossEntropyLoss()
    return modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, None, valid_dataloader, test_dataloader, loss_func, titles, sections, title2sections, sec2id, bm25_title, bm25_section, tokenizer

def test(modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, dataloader, loss_func):
    with torch.no_grad():
        modelp.eval()
        models.eval()
        total_loss = []
        eval_ans = []
        tp = 0
        total = 0
        for step, (querys, titles, sections, infer_titles, annotations_ids) in tqdm(enumerate(dataloader)):
            dis_final, lossp = modelp(querys, titles)
            dis_final, losss = models(querys, sections)
            rs2 = modelp.infer(querys, infer_titles)
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
                    if infer_titles[bid][cid] == titles[bid][0]:
                        count += 1
                    temp2 += title2sections[infer_titles[bid][cid]]
                    temp3 += [nid for x in title2sections[infer_titles[bid][cid]]]
                temp2_id = []
                tp += count
                if count == 0:
                    print('Failed Examples:')
                    print(query)
                    print(titles[bid][0])
                    print(infer_titles[bid])
                    print(temp)
                    print('------------------------------------')
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
            rs_scores = models.infer(querys, infer_section_candidates_pured)
            scores = scores_title * rs_scores
            rs2 = torch.topk(scores, config.infer_section_select, dim=1)
            scores = rs2[0]
            reference = []
            inds_sec = rs2[1].cpu().numpy()
            for bid in range(len(inds_sec)):
                temp = []
                for indc in inds_sec[bid]:
                    temp.append(infer_section_candidates_pured[bid][indc][0:config.maxium_sec])
                temp = ' [SEP] '.join(temp)
                reference.append(temp[0:1000])
            loss = 0.1*(lossp.mean() + losss.mean())
            total_loss.append(loss.item())
        modelp.train()
        models.train()
        print('accuracy: %f' %(tp/total))
        return np.array(total_loss).mean(), eval_ans


modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, train_dataloader, valid_dataloader, test_dataloader, loss_func, titles, sections, title2sections, sec2id, bm25_title, bm25_section, tokenizer = build(config)
test(modelp, models, model, optimizer_p, optimizer_s, optimizer_decoder, valid_dataloader, loss_func)