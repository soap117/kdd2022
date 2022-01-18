from models.data_process import read_clean_data
keys, titles, sections, title2sections, sec2id = read_clean_data('./data')
from rank_bm25 import BM25Okapi
import jieba
import numpy as np
corpus = sections
tokenized_corpus = [jieba.lcut(doc) for doc in corpus]
bm25_section = BM25Okapi(tokenized_corpus)

corpus = titles
tokenized_corpus = [jieba.lcut(doc) for doc in corpus]
bm25_title = BM25Okapi(tokenized_corpus)