import torch
import torch.nn as nn
from transformers import BertTokenizer
from pytorch_metric_learning import losses
from pytorch_metric_learning import distances

def info_nec(pos_pairs, neg_pairs):
    #max_val = torch.max(
    #    pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0]
    #).detach()
    numerator = torch.exp(pos_pairs).squeeze(1)
    denominator = torch.sum(torch.exp(neg_pairs), dim=1) + numerator
    log_exp = -torch.log((numerator / denominator))
    return log_exp

class TitleEncoder(nn.Module):
    def __init__(self, config):
        super(TitleEncoder, self).__init__()
        self.tokenizer = config.title_tokenizer
        filter_size = 3
        self.device = config.device
        self.conv = nn.ModuleList()
        self.embed = nn.Embedding(self.tokenizer.vocab_size, config.title_emb_dim)
        self.trans_layer = nn.Linear(config.title_emb_dim, config.title_emb_dim)
        self.final_activation = nn.Tanh()
        for l in range(2):
            tmp = nn.Conv1d(config.title_emb_dim, config.title_emb_dim, kernel_size=filter_size,
                            padding=int(filter_size - 1))
            self.conv.add_module('baseconv_%d' % l, tmp)
            tmp = nn.ReLU()
            self.conv.add_module('ReLU_%d' % l, tmp)
    def forward(self, title):
        B = len(title)
        L = len(title[0])
        title_new = []
        for one in title:
            title_new += one
        es = self.tokenizer(title_new, return_tensors='pt', padding=True, truncation=True).to(self.device)
        x = es['input_ids']
        x = self.embed(x)
        x = x.transpose(1, 2)
        tmp = x
        for idx, md in enumerate(self.conv):
            tmp = md(tmp)
        x = tmp
        x, _ = torch.max(x, dim=2)
        x = self.trans_layer(x)
        x = self.final_activation(x)
        x = x.view(B, L, -1)
        return x

    def query_forward(self, key):
        es = self.tokenizer(key, return_tensors='pt', padding=True, truncation=True).to(self.device)
        x = es['input_ids']
        x = self.embed(x)
        x = x.transpose(1, 2)
        tmp = x
        for idx, md in enumerate(self.conv):
            tmp = md(tmp)
        x = tmp
        x, _ = torch.max(x, dim=2)
        x = self.trans_layer(x)
        x = self.final_activation(x)
        return x

class ContextEncoder(nn.Module):
    def __init__(self, config):
        super(ContextEncoder, self).__init__()
        self.tokenizer = config.title_tokenizer
        filter_size = 3
        self.device = config.device
        self.conv = nn.ModuleList()
        self.embed = nn.Embedding(self.tokenizer.vocab_size, config.title_emb_dim)
        self.trans_layer = nn.Linear(config.title_emb_dim, config.title_emb_dim)
        self.final_activation = nn.Tanh()
        for l in range(4):
            tmp = nn.Conv1d(config.title_emb_dim, config.title_emb_dim, kernel_size=filter_size,
                            padding=int(filter_size - 1))
            self.conv.add_module('baseconv_%d' % l, tmp)
            tmp = nn.ReLU()
            self.conv.add_module('ReLU_%d' % l, tmp)
            if l <= 2:
                tmp = nn.MaxPool1d(2, 2)
                self.conv.add_module('maxpool_%d' % l, tmp)

    def forward(self, title):
        es = self.tokenizer(title, return_tensors='pt', padding=True, truncation=True).to(self.device)
        x_ = es['input_ids']
        x = self.embed(x_)
        x = x.transpose(1, 2)
        tmp = x
        for idx, md in enumerate(self.conv):
            tmp = md(tmp)
        x = tmp
        x, _ = torch.max(x, dim=2)
        x = self.trans_layer(x)
        x = self.final_activation(x)
        return x

class SecEncoder(nn.Module):
    def __init__(self, config):
        super(SecEncoder, self).__init__()
        self.tokenizer = config.title_tokenizer
        filter_size = 3
        self.device = config.device
        self.conv = nn.ModuleList()
        self.embed = nn.Embedding(self.tokenizer.vocab_size, config.context_emb_dim)
        self.trans_layer = nn.Linear(config.context_emb_dim, config.title_emb_dim)
        self.trans_layer_score = nn.Linear(config.context_emb_dim, 1)
        self.final_activation = nn.Tanh()
        self.drop_layer = nn.Dropout(0.25)
        for l in range(4):
            tmp = nn.Conv1d(config.context_emb_dim, config.context_emb_dim, kernel_size=filter_size,
                            padding=int(filter_size - 1))
            self.conv.add_module('baseconv_%d' % l, tmp)
            tmp = nn.ReLU()
            self.conv.add_module('ReLU_%d' % l, tmp)
            if l <= 1:
                tmp = nn.MaxPool1d(2, 2)
                self.conv.add_module('maxpool_%d' % l, tmp)
    def forward(self, title):
        B = len(title)
        L = len(title[0])
        title_new = []
        for one in title:
            title_new += one
        es = self.tokenizer(title_new, return_tensors='pt', padding=True, truncation=True).to(self.device)
        x_ = es['input_ids']
        x = self.embed(x_)
        x = x.transpose(1, 2)
        tmp = x
        for idx, md in enumerate(self.conv):
            tmp = md(tmp)
        x = tmp
        x, _ = torch.max(x, dim=2)
        score_context = self.trans_layer_score(x)
        x = self.trans_layer(x)
        x = self.final_activation(x)
        score_context = self.final_activation(self.drop_layer(score_context))
        x = x.view(B, L, -1)
        score_context = score_context.view(B, L)
        return x, score_context

class PageRanker(nn.Module):

    def __init__(self, config, title_encoder):
        super(PageRanker, self).__init__()
        self.query_encoder = title_encoder
        self.context_encoder = ContextEncoder(config)
        self.candidate_encoder = title_encoder
        self.loss_func = info_nec
        self.dis_func = distances.CosineSimilarity()
        self.drop_layer = torch.nn.Dropout(0.25)

    def forward(self, query, context, candidates):
        # query:[B,D] candidates:[B,L,D]
        query_embedding = self.query_encoder.query_forward(query)
        context_embedding = self.context_encoder(context)
        query_embedding = self.drop_layer(query_embedding*context_embedding)
        condidate_embeddings = self.drop_layer(self.candidate_encoder(candidates))
        dis_final = []
        for k in range(len(query_embedding)):
            temp_dis = self.dis_func(query_embedding[k].unsqueeze(0), condidate_embeddings[k])
            dis_final.append(temp_dis)
        dis_final = torch.cat(dis_final, 0)
        p_dis = dis_final[:, 0].unsqueeze(1)
        n_dis = dis_final[:, 1:]
        loss = self.loss_func(p_dis, n_dis)
        return dis_final, loss, query_embedding

    def infer(self, query_embedding, candidates):
        # query:[B,D] candidates:[B,L,D]
        #query_embedding = self.query_encoder.query_forward(query)
        condidate_embeddings = self.candidate_encoder(candidates)
        dis_final = []
        for k in range(len(query_embedding)):
            temp_dis = self.dis_func(query_embedding[k].unsqueeze(0), condidate_embeddings[k])
            dis_final.append(temp_dis)
        dis_final = torch.cat(dis_final, 0)

        return dis_final

class SectionRanker(nn.Module):

    def __init__(self, config, title_encoder):
        super(SectionRanker, self).__init__()
        self.query_encoder = title_encoder
        self.candidate_encoder = SecEncoder(config)
        self.loss_func = info_nec
        self.dis_func = distances.CosineSimilarity()
        self.drop_layer = torch.nn.Dropout(0.25)

    def forward(self, query_embedding, candidates):
        # query:[B,D] candidates:[B,L,D]
        #query_embedding = self.drop_layer(self.query_encoder.query_forward(query))
        condidate_embeddings, candidate_scores = self.candidate_encoder(candidates)
        condidate_embeddings = self.drop_layer(condidate_embeddings)
        dis_final = []
        for k in range(len(query_embedding)):
            temp_dis = self.dis_func(query_embedding[k].unsqueeze(0), condidate_embeddings[k])
            dis_final.append(temp_dis)
        dis_final = torch.cat(dis_final, 0) + candidate_scores
        p_dis = dis_final[:, 0].unsqueeze(1)
        n_dis = dis_final[:, 1:]
        loss = self.loss_func(p_dis, n_dis)
        return dis_final, loss

    def infer(self, query_embedding, candidates):
        # query:[B,D] candidates:[B,L,D]
        #query_embedding = self.drop_layer(self.query_encoder.query_forward(query))
        condidate_embeddings, candidate_scores = self.candidate_encoder(candidates)
        dis_final = []
        for k in range(len(query_embedding)):
            temp_dis = self.dis_func(query_embedding[k].unsqueeze(0), condidate_embeddings[k])
            dis_final.append(temp_dis)
        dis_final = torch.cat(dis_final, 0) + candidate_scores

        return dis_final