import torch
from transformers import BertTokenizer,BartTokenizer
from models.modeling_gpt2_att import GPT2LMHeadModel
from models.modeling_bart_ex import BartForConditionalGeneration as BartEX
from models.modeling_bart_ex import BartForAnnotation as BartAN
import jieba as sjieba

def pre_cut(text):
    temp = ' '.join(sjieba.lcut(text))
    temp = temp.replace('[ MASK ]', '[MASK]')
    temp = temp.replace('[ CLS ]', '[CLS]')
    temp = temp.replace('[ SEP ]', '[SEP]')
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
    for fv in function_vocabs:
        sjieba.add_word(fv)
    wait_list = original_vocabs[106:]
    new_vocabs = function_vocabs+target_vocabs[0:14697]
    c = 0
    while len(new_vocabs)<30000:
        temp = wait_list[c].replace('\n', '')
        if temp in new_vocabs:
            c += 1
        else:
            new_vocabs.append(temp)
    with open('./tokenizer/vocab.txt', 'w', encoding='utf-8') as f:
        f.writelines([x+'\n' for x in new_vocabs])
    tokenizer = BertTokenizer.from_pretrained('./tokenizer', do_lower_case=False)
    tokenizer.is_pretokenized = True
    tokenizer.tokenize_chinese_chars = True
    tokenizer.do_basic_tokenize = False
    return tokenizer

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
        self.title_tokenizer.model_max_length = 768
        self.key_tokenizer = self.title_tokenizer
        self.tokenizer_editplus = modify_tokenizer()
        self.tokenizer_editplus.add_special_tokens(
            {'additional_special_tokens': ['[unused1]', '[unused2]', '[unused3]', '[unused4]']})
        self.pre_cut = pre_cut
        self.modeld = BartAN
        self.modeld_sen = BartEX
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
        self.hidden_anno_len = 10
        self.hidden_anno_len_rnn = 10
        self.para_hidden_len = 10
        self.multi_gpu = False
        self.full_gpu_id = 0
        self.max_query = 12
        self.data_file_old = 'data/mydata_v5.pkl'
        self.data_file_anno_old = 'data/mydata_v5_anno.pkl'
        self.data_file = 'data/mydata_v5_para.pkl'
        self.data_file_anno = 'data/mydata_v5_anno.pkl'

config = Config(16)