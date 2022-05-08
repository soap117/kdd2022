import re, pickle, numpy
def creat_sentence(data_new):
    sentence_format = {}
    for file in data_new:
        anno = file['file']
        src = anno['src']
        src = re.sub('（(.*。)）', '', src)
        src = re.sub('\*\*', '', src)
        src = src.replace('(', '（')
        src = src.replace(')', '）')
        src = src.replace('\n', '').replace('。。', '。')
        tar = anno['tar']
        tar = re.sub('（(.*。)）', '', tar)
        tar = re.sub('\*\*', '', tar)
        tar = tar.replace('\n', '').replace('。。', '。')
        tar = tar.replace('(', '（')
        tar = tar.replace(')', '）')
        file['origin_key'] = file['origin_key'].replace('(', '（')
        file['origin_key'] = file['origin_key'].replace(')', '）')
        file['key'] = file['key'].replace('(', '（')
        file['key'] = file['key'].replace(')', '）')
        if src[-1] == '。' and tar[-1] != '。':
            tar += '。'
        if tar[-1] == '。' and src[-1] != '。':
            src += '。'
        data_key = None
        src_sts = src.split('。')
        tar_sts = tar.split('。')
        for i in range(len(src_sts)-1, -1, -1):
            if len(src_sts[i]) == 0:
                del src_sts[i]
        for i in range(len(tar_sts)-1, -1, -1):
            if len(tar_sts[i]) == 0:
                del tar_sts[i]
        dt = len(tar_sts) - len(src_sts)
        if dt == 0:
            for src_st, tar_st in zip(src_sts, tar_sts):
                if file['original_key'] in src_st and src_st != tar_st:
                    file['src_st'] = src_st
                    file['tar_st'] = tar_st
                    pos = re.search(file['original_key'], src_st)
                    if pos is not None:
                        file['position'] = pos.regs
                        data_key = {'key': file['key'], 'origin': file['origin_key'], 'anno': file['anno'], 'urls': file['urls'], 'rsecs': file['rsecs'],
                                    'rpsecs': file['rpsecs'], 'pos': file['position']}
                    break
        else:
            print('miss')
        if data_key is not None:
            file_sen = file['file']['textid'] + file['src_st']
            if file_sen in sentence_format:
                sentence_format[file_sen]['data'].append(data_key)
            else:
                sentence_format[file_sen] = {}
                sentence_format[file_sen]['data'] = [data_key]
                sentence_format[file_sen]['src_st'] = file['src_st']
                sentence_format[file_sen]['tar_st'] = file['tar_st']
                sentence_format[file_sen]['textid'] = file['file']['textid']
    return list(sentence_format.values())
with open('mydata_new_clean_v4_sec_sub_trn.pkl', 'rb') as f:
    data_new = pickle.load(f)
sentence_format = creat_sentence(data_new)
with open('mydata_sen_clean_v4_sec_sub_trn.pkl', 'wb') as f:
    pickle.dump(sentence_format, f)