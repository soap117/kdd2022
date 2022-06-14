import torch
from transformers import BertTokenizer,BartTokenizer
from models.modeling_gpt2_att import GPT2LMHeadModel
from models.modeling_bart_ex import BartForConditionalGeneration as BartEX
from models.modeling_bart_ex import BartForAnnotation as BartAN
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
        self.modeld = BartAN
        self.modeld_sen = BartEX
        self.modeld_ann = BartAN
        self.title_emb_dim = 128
        self.key_emb_dim = 128
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
        self.maxium_sec = 512
        self.hidden_anno_len = 20
        self.hidden_anno_len_rnn = 20
        self.multi_gpu = False
        self.full_gpu_id = 0
        self.data_file_old = 'data/mydata_v5_para.pkl'
        self.data_file_anno_old = 'data/mydata_v5_anno.pkl'
        self.data_file = 'data/mydata_v5_para.pkl'
        self.data_file_anno = 'data/mydata_v5_anno.pkl'

config = Config(16)