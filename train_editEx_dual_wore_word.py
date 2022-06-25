import os
import cuda
import torch
import torch.nn as nn
from config import Config
config = Config(8)
from models.units_sen_editEX import MyData, get_decoder_att_map, mask_ref, read_clean_data, find_spot_para, find_UNK, operation2sentence_word
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
    if not debug_flag and os.path.exists(config.data_file.replace('.pkl', '_train_dataset_edit_purewo_word.pkl')):
        train_dataset = torch.load(config.data_file.replace('.pkl', '_train_dataset_edit_purewo_word.pkl'))
        valid_dataset = torch.load(config.data_file.replace('.pkl', '_valid_dataset_edit_purewo_word.pkl'))
        test_dataset = torch.load(config.data_file.replace('.pkl', '_test_dataset_edit_purewo_word.pkl'))
    else:
        train_dataset = MyData(config, tokenizer, config.data_file.replace('.pkl', '_train_dataset_raw.pkl'), titles, sections, title2sections, sec2id,
                               bm25_title, bm25_section, is_pure=True, wo_re=True, word=True)
        valid_dataset = MyData(config, tokenizer, config.data_file.replace('.pkl', '_valid_dataset_raw.pkl'), titles, sections, title2sections, sec2id,
                               bm25_title,
                               bm25_section, is_pure=True, wo_re=True, word=True)
        test_dataset = MyData(config, tokenizer, config.data_file.replace('.pkl', '_test_dataset_raw.pkl'), titles, sections, title2sections, sec2id, bm25_title,
                              bm25_section, is_pure=True, wo_re=True, word=True)
        torch.save(train_dataset, config.data_file.replace('.pkl', '_train_dataset_edit_purewo_word.pkl'))
        torch.save(valid_dataset, config.data_file.replace('.pkl', '_valid_dataset_edit_purewo_word.pkl'))
        torch.save(test_dataset, config.data_file.replace('.pkl', '_test_dataset_edit_purewo_word.pkl'))

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size
                                  , collate_fn=train_dataset.collate_fn)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=1
                                  , collate_fn=train_dataset.collate_fn_test)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1
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
    modele.load_state_dict(save_data['model'], strict=False)
    print('Load pretrained E')
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
    modelp.cuda()
    models.cuda()
    modele.cuda()
    modeld.cuda()
    modelp.train()
    models.train()
    modele.train()
    modeld.train()
    optimizer_p = AdamW(modelp.parameters(), lr=config.lr*0.1)
    optimizer_s = AdamW(models.parameters(), lr=config.lr*0.1)
    optimizer_encoder = AdamW(modele.parameters(), lr=config.lr*0.01 )
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
    for epoch in range(config.train_epoch*4):
        torch.cuda.empty_cache()
        for step, (querys, querys_ori, querys_context, titles, sections, infer_titles, src_sens, src_sens_ori, tar_sens, cut_list, edit_sens) in zip(
                tqdm(range(data_size)), train_dataloader):
            decoder_inputs = tokenizer(src_sens, return_tensors="pt", padding=True, truncation=True)
            decoder_ids = decoder_inputs['input_ids'].cuda()

            decoder_anno_position = []
            hidden_annotation = None

            target_ids = tokenizer(tar_sens, return_tensors="pt", padding=True, truncation=True)['input_ids'].to(
                config.device)
            decoder_inputs_ori = tokenizer(src_sens_ori, return_tensors="pt", padding=True, truncation=True)
            decoder_ids_ori = decoder_inputs_ori['input_ids'].to(config.device)
            edit_sens_token = [['[CLS]'] + x + ['[SEP]'] for x in edit_sens]
            edit_sens_token_ids = [torch.LongTensor(tokenizer.convert_tokens_to_ids(x)) for x in edit_sens_token]
            edit_sens_token_ids = pad_sequence(edit_sens_token_ids, batch_first=True, padding_value=0).to(config.device)

            # Clean actions
            input_actions = torch.zeros_like(edit_sens_token_ids) + 5
            input_actions = torch.where(
                (edit_sens_token_ids == 1) | (edit_sens_token_ids == 2) | (edit_sens_token_ids == 101) | (
                            edit_sens_token_ids == 102) | (
                            edit_sens_token_ids == 0), edit_sens_token_ids,
                input_actions)

            #decoder_edits= tokenizer(edit_sens, return_tensors="pt", padding=True, truncation=True)
            #decoder_ids_edits = decoder_edits['input_ids'].to(config.device)

            logits_action, logits_edit, hidden_edits = modeld(input_ids=decoder_ids, decoder_input_ids=target_ids,
                             anno_position=decoder_anno_position, hidden_annotation=hidden_annotation, input_edits=edit_sens_token_ids, input_actions=input_actions, org_ids=decoder_ids_ori)
            targets_edit = edit_sens_token_ids[:, 1:]
            min_len = min(targets_edit.shape[1], logits_edit.shape[1])
            targets_edit = targets_edit[:, 0:min_len]
            logits_edit = logits_edit[:, 0:min_len]
            _, predictions_edit = torch.max(logits_edit, dim=-1)
            targets_action = input_actions[:, 1:]
            targets_action = targets_action[:, 0:min_len]
            logits_action = logits_action[:, 0:min_len]
            _, predictions_action = torch.max(logits_action, dim=-1)
            predictions = torch.where(predictions_action != 5, predictions_action, predictions_edit)

            results = tokenizer.batch_decode(predictions)
            results = [tokenizer.convert_tokens_to_string(x) for x in results]
            results = [x.replace(' ', '') for x in results]
            results = [x.replace('[PAD]', '') for x in results]
            results = [x.replace('[unused1]', '[K]') for x in results]
            results = [x.replace('[unused2]', '[D]') for x in results]
            results = [x.replace('[CLS]', '') for x in results]
            results = [x.split('[SEP]')[0] for x in results]

            logits_action = logits_action.reshape(-1, logits_action.shape[2])
            tar_lens = targets_action.ne(0).sum(1).float()
            targets_flat = targets_action.reshape(-1).to(config.device)
            #masks = torch.ones_like(targets)
            #masks[torch.where(targets == 0)] = 0
            lossd_ac = loss_func(logits_action, targets_flat)
            lossd_ac[targets_flat == 0] = 0
            lossd_ac = lossd_ac.view(targets_action.size())
            lossd_ac = lossd_ac.sum(1).float()
            lossd_ac = lossd_ac/tar_lens
            lossd_ac = lossd_ac.mean()

            logits_edit = logits_edit.reshape(-1, logits_edit.shape[2])
            tar_lens = ((targets_edit!=0)&(targets_edit!=1)&(targets_edit!=2)&(targets_edit!=101)&(targets_edit!=102)).sum(1).float()+1e-5
            targets_flat = targets_edit.reshape(-1).to(config.device)
            # masks = torch.ones_like(targets)
            # masks[torch.where(targets == 0)] = 0
            lossd_ed = loss_func(logits_edit, targets_flat)
            lossd_ed[(targets_flat==0)|(targets_flat==1)|(targets_flat==2)|(targets_flat==101)|(targets_flat==102)|(targets_flat==100)] = 0
            lossd_ed = lossd_ed.view(targets_edit.size())
            lossd_ed = lossd_ed.sum(1).float()
            lossd_ed = lossd_ed / tar_lens
            lossd_ed = lossd_ed.mean()
            loss = lossd_ac + lossd_ed
            optimizer_p.zero_grad()
            optimizer_s.zero_grad()
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            loss.backward()
            if epoch > 3:
                optimizer_p.step()
                optimizer_s.step()
            optimizer_encoder.step()
            optimizer_decoder.step()
            if step%400 == 0:
                print('loss P:%f loss S:%f loss AC:%f loss ED:%f' %(0, 0, lossd_ac.item(), lossd_ed.item()))
                print(results[0:2])
                print('---------------------------')
        test_loss, eval_ans, grand_ans = test(modelp, models, modele, modeld, valid_dataloader, loss_func)
        d_eval_loss = test_loss
        if d_eval_loss <= min_loss_d:
            print(count_p, count_s)
            print('update-all')
            print('New Test Loss D:%f' % (d_eval_loss))
            state = {'epoch': epoch, 'config': config, 'models': models.state_dict(), 'modelp': modelp.state_dict(), 'modele': modele.state_dict(), 'modeld': modeld.state_dict(),
                     'eval_rs': eval_ans}
            torch.save(state, './results/' + config.data_file.replace('.pkl', '_models_edit_dual_wore.pkl').replace('data/', ''))
            min_loss_d = d_eval_loss
            for one, one_g in zip(eval_ans[0:5], grand_ans[0:5]):
                print(one)
                print(one_g)
            print('+++++++++++++++++++++++++++++++')
        else:
            print(count_p, count_s)
            print('New Larger Test Loss D:%f' % (d_eval_loss))
            for one, one_g in zip(eval_ans[0:5], grand_ans[0:5]):
                print(one)
                print(one_g)
            print('+++++++++++++++++++++++++++++++')
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
            decoder_inputs = tokenizer(src_sens, return_tensors="pt", padding=True, truncation=True)
            decoder_ids = decoder_inputs['input_ids'].cuda()

            decoder_anno_position = []
            hidden_annotation = None

            target_ids = tokenizer(tar_sens, return_tensors="pt", padding=True, truncation=True)['input_ids'].to(
                config.device)
            decoder_inputs_ori = tokenizer(src_sens_ori, return_tensors="pt", padding=True, truncation=True)
            decoder_ids_ori = decoder_inputs_ori['input_ids'].to(config.device)
            edit_sens_token = [['[CLS]'] + x + ['[SEP]'] for x in edit_sens]
            edit_sens_token_ids = [torch.LongTensor(tokenizer.convert_tokens_to_ids(x)) for x in edit_sens_token]
            edit_sens_token_ids = pad_sequence(edit_sens_token_ids, batch_first=True, padding_value=0).to(config.device)

            # Clean actions
            input_actions = torch.zeros_like(edit_sens_token_ids) + 5
            input_actions = torch.where(
                (edit_sens_token_ids == 1) | (edit_sens_token_ids == 2) | (edit_sens_token_ids == 101) | (
                        edit_sens_token_ids == 102) | (
                        edit_sens_token_ids == 0), edit_sens_token_ids,
                input_actions)

            # decoder_edits= tokenizer(edit_sens, return_tensors="pt", padding=True, truncation=True)
            # decoder_ids_edits = decoder_edits['input_ids'].to(config.device)
            logits_action, logits_edit, hidden_edits = modeld(input_ids=decoder_ids, decoder_input_ids=target_ids,
                                          anno_position=decoder_anno_position, hidden_annotation=hidden_annotation,
                                          input_edits=edit_sens_token_ids, input_actions=input_actions, org_ids=decoder_ids_ori, force_ratio=0.0, eval=True)

            targets = target_ids[:, 1:]
            _, action_predictions = torch.max(logits_action, dim=-1)
            _, edit_predictions = torch.max(logits_edit, dim=-1)
            predictions = torch.where(action_predictions != 5, action_predictions, edit_predictions)
            tokenized_ori = [find_UNK(x, tokenizer.tokenize(x), tokenizer) for x in src_sens]
            predictions, predictions_text = operation2sentence_word(predictions, decoder_ids, tokenized_ori, tokenizer)
            results = predictions_text
            results = [x.replace('[PAD]', '') for x in results]
            results = [x.replace('[CLS]', '') for x in results]
            results = [x.replace('[MASK]', '') for x in results]
            results = [x.split('[SEP]')[0] for x in results]
            ground_truth = tar_sens
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
        return -bleu_scores, eval_ans, eval_gt

modelp, models, modele, modeld, optimizer_p, optimizer_s, optimizer_encoder, optimizer_decoder, train_dataloader, valid_dataloader, test_dataloader, loss_func, titles, sections, title2sections, sec2id, bm25_title, bm25_section, tokenizer = build(config)
state = train_eval(modelp, models, modele, modeld, optimizer_p, optimizer_s, optimizer_encoder, optimizer_decoder, train_dataloader, valid_dataloader, loss_func)