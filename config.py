import torch
from transformers import BertTokenizer
class Config(object):

    """配置参数"""
    def __init__(self, batch_size):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.title_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.title_tokenizer.model_max_length = 300
        self.key_tokenizer= self.title_tokenizer
        self.title_emb_dim = 128
        self.key_emb_dim = 128
        self.context_emb_dim = 256
        self.lr = 5e-4
        self.train_epoch = 100
        self.data_path = './data/mydata.pkl'
        self.neg_num = 9
        self.neg_strong_num = 3
        self.infer_title_range = 10
        self.infer_title_select = 3
        self.batch_size = batch_size
        self.infer_section_range = 10
        self.infer_section_select = 3
        self.maxium_sec = 300
        self.data_file = 'data/mydata_new_clean_v4_sec_sub_trn.pkl'

config = Config(16)