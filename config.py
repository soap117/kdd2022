import torch
from transformers import BertTokenizer,BartTokenizer
from models.modeling_gpt2_att import GPT2LMHeadModel
from models.modeling_bart_ex import BartForConditionalGeneration as BartEX
from models.modeling_bart_ex import BartForAnnotation as BartAN
import jieba as sjieba
import numpy as np

def pre_cut(text):
    temp = ' '.join(sjieba.lcut(text.lower()))
    temp = temp.replace('[ mask ]', '[MASK]')
    temp = temp.replace('[ cls ]', '[CLS]')
    temp = temp.replace('[ sep ]', '[SEP]')
    temp = temp.replace('[ unused1 ]', '[unused1]')
    temp = temp.replace('[ unused2 ]', '[unused2]')
    temp = temp.replace('[ unused3 ]', '[unused3]')
    temp = temp.replace('[ unused4 ]', '[unused4]')
    return temp

def modify_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('fnlp/bart-base-chinese')
    tokenizer.save_pretrained('./tokenizer')
    original_vocabs = open('./tokenizer/vocab.txt', 'r', encoding='utf-8').readlines()
    original_vocabs = [x.replace('\n', '') for x in original_vocabs]
    target_vocabs = open('./tokenizer/vocab_ch.txt', 'r', encoding='utf-8').readlines()
    target_vocabs = [x.split()[0] for x in target_vocabs if x !='\n']
    function_vocabs = original_vocabs[0:106]
    function_vocabs[10] = '$'
    for fv in function_vocabs:
        sjieba.add_word(fv)
    new_vocabs = function_vocabs
    for new_word in target_vocabs[0:30000]:
        if new_word not in new_vocabs:
            new_vocabs.append(new_word)
        else:
            print(new_word)
    with open('./tokenizer/vocab.txt', 'w', encoding='utf-8') as f:
        f.writelines([x+'\n' for x in new_vocabs])
    tokenizer = BertTokenizer.from_pretrained('./tokenizer', do_lower_case=False)
    tokenizer.is_pretokenized = True
    tokenizer.tokenize_chinese_chars = True
    tokenizer.do_basic_tokenize = False
    print("Loading Glove embeddings")
    embed_size = 300
    with open('./tokenizer/sgns.wiki.word', 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        model = {}
        w_set = set(tokenizer.vocab)
        embedding_matrix = np.zeros(shape=(len(tokenizer.vocab), embed_size))

        for line in lines:
            splitLine = line.split()
            word = splitLine[0]
            if word in w_set:  # only extract embeddings in the word_list
                embedding = np.array([float(val) for val in splitLine[1:]])
                model[word] = embedding
                embedding_matrix[tokenizer.vocab[word]] = embedding
                # if len(model) % 1000 == 0:
                    # print("processed %d vocab_data" % len(model))
    print("%d words out of %d has embeddings in the glove file" % (len(model), len(tokenizer.vocab)))
    embedding_matrix = torch.FloatTensor(embedding_matrix)
    return tokenizer, embedding_matrix

class Config(object):

    """配置参数"""
    def __init__(self, batch_size):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备bert_model = 'facebook/bart-base'
        self.bert_model = 'fnlp/bart-base-chinese'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[unused1]','[unused2]','[unused3]','[unused4]']})
        self.CLS = self.tokenizer.cls_token_id
        self.SEP = self.tokenizer.sep_token_id
        self.title_tokenizer = self.tokenizer
        self.title_tokenizer.model_max_length = 512
        self.key_tokenizer = self.title_tokenizer
        #tokenizer_editplus, embedding_new = modify_tokenizer()
        #self.tokenizer_editplus = tokenizer_editplus
        #self.embedding_new = embedding_new
        #self.tokenizer_editplus.add_special_tokens(
        #    {'additional_special_tokens': ['[unused1]', '[unused2]', '[unused3]', '[unused4]']})
        #self.tokenizer_editplus.model_max_length = 512
        self.pre_cut = pre_cut
        self.modeld = BartAN
        self.modeld_sen = BartEX
        self.drop_rate = 0.2
        self.modeld_ann = BartAN
        self.title_emb_dim = 128
        self.key_emb_dim = 128
        self.rnn_layer = 1
        self.rnn_dim = 400
        self.context_emb_dim = 256
        self.lr = 1e-3
        self.train_epoch = 20
        self.neg_num = 9
        self.neg_strong_num = 3
        self.infer_title_range = 10
        self.infer_title_select = 3
        self.batch_size = batch_size
        self.infer_section_range = 10
        self.infer_section_select = 3
        self.maxium_sec = 100
        self.hidden_anno_len = 30
        self.multi_gpu = False
        self.pure = True
        self.full_gpu_id = 0
        self.max_query = 12
        self.data_file_old = 'data/mydata_v5.pkl'
        self.data_file_anno_old = 'data/mydata_v5_anno.pkl'
        self.data_file = 'data/mydata_v5_para.pkl'
        self.data_file_anno = 'data/mydata_v5_anno.pkl'

config = Config(16)