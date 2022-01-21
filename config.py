import torch
from transformers import BertTokenizer
class Config(object):

    """配置参数"""
    def __init__(self, batch_size):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.title_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.gpu_id = 2
        try:
            torch.cuda.set_device(self.gpu_id)
        except:
            torch.cuda.set_device(0)
        self.key_tokenizer= self.title_tokenizer
        self.title_emb_dim = 64
        self.key_emb_dim = 64
        self.batch_size = 2
        self.lr = 1e-4
        self.train_epoch = 10
        self.data_path = './data/mydata.pkl'
        self.neg_num = 5
        self.infer_title_range = 10
        self.infer_title_select = 3
        self.batch_size = batch_size
        self.infer_section_range = 10
        self.infer_section_select = 3
config = Config(2)