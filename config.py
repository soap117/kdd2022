import torch
from transformers import BertTokenizer
class Config(object):

    """配置参数"""
    def __init__(self, batch_size):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.title_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.key_tokenizer= self.title_tokenizer
        self.title_emb_dim = 64
        self.key_emb_dim = 64
        self.batch_size = 2
        self.lr = 1e-4
        self.train_epoch = 4
        self.data_path = './data/mydata.pkl'
config = Config(2)