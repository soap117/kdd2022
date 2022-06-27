import os
import pickle
import jieba
import nltk
import numpy as np
import pandas as pd
import data
from tqdm import tqdm
from nltk import pos_tag
from label_edits import sent2edit
import jieba.posseg as posseg
# This script contains the reimplementation of the pre-process steps of the dataset
# For the editNTS system to run, the dataset need to be in a pandas DataFrame format
# with columns ['comp_tokens', 'simp_tokens','comp_ids','simp_ids', 'comp_pos_tags', 'comp_pos_ids', edit_labels','new_edit_ids']

PAD = 'PAD' #  This has a vocab id, which is used to represent out-of-vocabulary words [0]
UNK = 'UNK' #  This has a vocab id, which is used to represent out-of-vocabulary words [1]
KEEP = 'KEEP' # This has a vocab id, which is used for copying from the source [2]
DEL = 'DEL' # This has a vocab id, which is used for deleting the corresponding word [3]
START = 'START' # this has a vocab id, which is uded for indicating start of the sentence for decoding [4]
STOP = 'STOP' # This has a vocab id, which is used to stop decoding [5]

def remove_lrb(sent_string):
    # sent_string = sent_string.lower()
    frac_list = sent_string.split('-lrb-')
    clean_list = []
    for phrase in frac_list:
        if '-rrb-' in phrase:
            clean_list.append(phrase.split('-rrb-')[1].strip())
        else:
            clean_list.append(phrase.strip())
    clean_sent_string =' '.join(clean_list)
    return clean_sent_string

def replace_lrb(sent_string):
    sent_string = sent_string.lower()
    # new_sent= sent_string.replace('-lrb-','(').replace('-rrb-',')')
    new_sent = sent_string.replace('-lrb-', '').replace('-rrb-', '')
    return new_sent


def process_raw_data(comp_txt, simp_txt, is_train=True):

    comp_txt_pos = []
    for line in tqdm(comp_txt):
        comp_txt_pos.append(list(posseg.cut(line)))
    simp_txt_pos = []
    for line in tqdm(simp_txt):
        simp_txt_pos.append(list(posseg.cut(line)))
    comp_txt = [[x.word for x in line] for line in comp_txt_pos]
    simp_txt = [[x.word for x in line] for line in simp_txt_pos]
    # df_comp = pd.read_csv('mydata/%s_comp.csv'%dataset,  sep='\t')
    # df_simp= pd.read_csv('mydata/%s_simp.csv'%dataset,  sep='\t')
    assert len(comp_txt) == len(simp_txt)
    df = pd.DataFrame(
                        {'comp_tokens': comp_txt,
                         'simp_tokens': simp_txt,
                        })
    def get_vocab(df):
        word_dict ={}
        comp_sentences = df['comp_tokens'].tolist()
        simp_sentences = df['simp_tokens'].tolist()
        for sen in comp_sentences:
            for word in sen:
                if word in word_dict:
                    word_dict[word] += 1
                else:
                    word_dict[word] = 1
        for sen in simp_sentences:
            for word in sen:
                if word in word_dict:
                    word_dict[word] += 1
                else:
                    word_dict[word] = 1
        list_word = word_dict.items()
        list_word = sorted(list_word, key=lambda x: x[1], reverse=True)
        list_word = [x[0] +' '+ str(x[1]) + '\n' for x in list_word]
        f = open('./vocab_data/vocab.txt', 'w', encoding='utf-8')
        f.writelines(list_word)
        f.close()
    def add_edits(df):
        """
        :param df: a Dataframe at least contains columns of ['comp_tokens', 'simp_tokens']
        :return: df: a df with an extra column of target edit operations
        """
        comp_sentences = df['comp_tokens'].tolist()
        simp_sentences = df['simp_tokens'].tolist()
        pair_sentences = list(zip(comp_sentences,simp_sentences))
        edits_list = []
        for l in tqdm(pair_sentences):
            edits_list.append(sent2edit(l[0],l[1]))
        df['edit_labels'] = edits_list
        return df
    def create_pos_tag_table(pos_sentences):
        pos_tag_dict = {'PAD':0, 'UNK':1, 'START':2, 'STOP':3}
        for sent in pos_sentences:
            for word in sent:
                if word.flag not in pos_tag_dict:
                    pos_tag_dict[word.flag] = len(pos_tag_dict)
        with open('./vocab_data/chn_postag_set.p', 'wb') as f:
            pickle.dump(pos_tag_dict, f)
        return pos_tag_dict
    def add_pos(df, pos_sentences):
        src_sentences = df['comp_tokens'].tolist()
        df['comp_pos_tags'] = pos_sentences
        pos_vocab = data.POSvocab('./vocab_data/')
        pos_ids_list = []
        for sent in pos_sentences:
            pos_ids = [pos_vocab.w2i[w.flag] if w.flag in pos_vocab.w2i.keys() else pos_vocab.w2i[UNK] for w in sent]
            pos_ids_list.append(pos_ids)
        df['comp_pos_ids'] = pos_ids_list
        return df
    if is_train:
        get_vocab(df)
        create_pos_tag_table(comp_txt_pos)
    df = add_pos(df, comp_txt_pos)
    df = add_edits(df)
    return df

def editnet_data_to_editnetID(df,output_path):
    """
    this function reads from df.columns=['comp_tokens', 'simp_tokens', 'edit_labels','comp_pos_tags','comp_pos_ids']
    and add vocab ids for comp_tokens, simp_tokens, and edit_labels
    :param df: df.columns=['comp_tokens', 'simp_tokens', 'edit_labels','comp_pos_tags','comp_pos_ids']
    :param output_path: the path to store the df
    :return: a dataframe with df.columns=['comp_tokens', 'simp_tokens', 'edit_labels',
                                            'comp_ids','simp_id','edit_ids',
                                            'comp_pos_tags','comp_pos_ids'])
    """
    out_list = []
    vocab = data.Vocab()
    vocab.add_vocab_from_file('./vocab_data/vocab.txt', 30000)

    def prepare_example(example, vocab):
        """
        :param example: one row in pandas dataframe with feild ['comp_tokens', 'simp_tokens', 'edit_labels']
        :param vocab: vocab object for translation
        :return: inp: original input sentence,
        """
        comp_id = np.array([vocab.w2i[i] if i in vocab.w2i.keys() else vocab.w2i[UNK] for i in example['comp_tokens']])
        simp_id = np.array([vocab.w2i[i] if i in vocab.w2i.keys() else vocab.w2i[UNK] for i in example['simp_tokens']])
        edit_id = np.array([vocab.w2i[i] if i in vocab.w2i.keys() else vocab.w2i[UNK] for i in example['edit_labels']])
        return comp_id, simp_id, edit_id  # add a dimension for batch, batch_size =1

    for i,example in df.iterrows():
        print(i)
        comp_id, simp_id, edit_id = prepare_example(example,vocab)
        ex=[example['comp_tokens'], comp_id,
         example['simp_tokens'], simp_id,
         example['edit_labels'], edit_id,
         example['comp_pos_tags'],example['comp_pos_ids']
         ]
        out_list.append(ex)
    outdf = pd.DataFrame(out_list, columns=['comp_tokens','comp_ids', 'simp_tokens','simp_ids',
                                            'edit_labels','new_edit_ids','comp_pos_tags','comp_pos_ids'])
    outdf.to_pickle(output_path)
    print('saved to %s'%output_path)



src_data = pickle.load(open('./mydata/train/src_txts.pkl', 'rb'))
tar_data = pickle.load(open('./mydata/train/tar_txts.pkl', 'rb'))
df = process_raw_data(src_data, tar_data)
editnet_data_to_editnetID(df, './mydata/train.df.filtered.pos')
src_data = pickle.load(open('./mydata/valid/src_txts.pkl', 'rb'))
tar_data = pickle.load(open('./mydata/valid/tar_txts.pkl', 'rb'))
df = process_raw_data(src_data, tar_data, False)
editnet_data_to_editnetID(df, './mydata/val.df.filtered.pos')
src_data = pickle.load(open('./mydata/test/src_txts.pkl', 'rb'))
tar_data = pickle.load(open('./mydata/test/tar_txts.pkl', 'rb'))
df = process_raw_data(src_data, tar_data, False)
editnet_data_to_editnetID(df, './mydata/test.df.filtered.pos')