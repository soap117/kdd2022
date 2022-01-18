import pickle
import re
import os

def read_clean_data(path):
    sample_data = pickle.load(open(path, 'rb'))
    keys = []
    titles = []
    sections = []
    title2sections = {}
    urls = set()
    sec2id = {}
    for one in sample_data:
        if len(one['urls']) > 0:
            for tid, (title, url) in enumerate(zip(one['rpsecs'], one['urls'])):
                if len(title) > 0:
                    web_title = title[-1]
                    web_title = re.sub('_.+', '', web_title)
                    web_title = re.sub(' -.+', '', web_title)
                    one['rpsecs'][tid][-1] = web_title
                    sections += title[0:-1]
                    titles.append(web_title)
                    if web_title in title2sections and url not in urls:
                        title2sections[web_title] += title[0:-1]
                        urls.add(url)
                    elif web_title not in title2sections:
                        title2sections[web_title] = title[0:-1]
                        urls.add(url)

            keys.append(one)
    titles = list(set(titles))
    sections = list(set(sections))
    for k in range(len(sections)-1, -1, -1):
        if len(sections[k]) < 60:
            del sections[k]
    for tid, temp in enumerate(sections):
        sec2id[temp] = tid
    return keys, titles, sections, title2sections, sec2id