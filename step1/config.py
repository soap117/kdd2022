# coding: UTF-8
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertConfig
import pickle

class Config(object):

    """配置参数"""
    def __init__(self, batch_size, bert_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 10                                            # epoch数
        self.batch_size = batch_size                                       # mini-batch大小
        self.pad_size = 512                                            # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-6                                      # 学习率
        self.warmup_proportion = 0.03

        self.bert_path = bert_path

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)

        self.hidden_size = 768
