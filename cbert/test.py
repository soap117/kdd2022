from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import pickle

with open('result/label_test_best.txt', 'r', encoding='utf-8') as f:
    data = f.read().split('\n\n')
    print(len(data))
    predictions = []
    true_labels = []
    predict_keywords = []
    label_keywords = []
    cnt = 0
    for sample in data:
        labels = []
        outputs = []
        if len(sample)<1: continue
        keyword = ""
        true_keyword = ""
        keywords = []
        true_keywords = []
        preoutput=prelabel=0
        for line in sample.split('\n'):
            if len(line)<2:continue
            t = line.split(' ')
            if t[0]=='[PAD]':continue
            cnt +=1
            label = t[-2]
            output = t[-1]
            label = ord(label)-ord('0')
            output = ord(output)-ord('0')

            if output == 1:
                if keyword!="":
                    keywords.append(keyword)
                keyword = t[0]
            elif output==3:
                if keyword!="":
                    keyword += t[0]
            elif output==4:
                if keyword!="":
                    keyword += t[0]
                    keywords.append(keyword)
                    keyword = ""
            elif output==0:
                if keyword!="":
                    keywords.append(keyword)
                    keyword=""

            if label == 1:
                if true_keyword != "":
                    true_keywords.append(true_keyword)
                true_keyword = t[0]
            elif label == 3:
                if true_keyword != "":
                    true_keyword += t[0]
            elif label == 4:
                if true_keyword != "":
                    true_keyword += t[0]
                    true_keywords.append(true_keyword)
                    true_keyword = ""
            elif label == 0:
                if true_keyword != "":
                    true_keywords.append(true_keyword)
                    true_keyword = ""
            preoutput = output
            prelabel = label

            labels.append(label)
            outputs.append(output)

        # print(len(labels))
        predictions.extend(outputs)
        true_labels.extend(labels)
        predict_keywords.append(keywords)
        label_keywords.append(true_keywords)
    print(cnt)
    print(predictions[:50])
    print(true_labels[:50])
    acc = accuracy_score(predictions, true_labels)
    precision = precision_score(predictions, true_labels, average='micro')
    recall_score = recall_score(predictions, true_labels, average='micro')
    f1 = f1_score(predictions, true_labels, average='micro')
    print('token level accuracy:',acc)
    print('token level f1:',f1)
    print('token level precision:',precision)
    print('token level recall_score:',recall_score)


keywords_all = label_keywords
correct = 0
p_n = 0
g_n = 0

# with open('./data/valid/keywordset_list_char_0.pkl','rb') as f:
#     _keywords_all = pickle.load(f)
#
# for u,v in zip(keywords_all, _keywords_all):
#     print(set(u))
#     print(set(v))
#     print()

for p_ks, y_ks in zip(predict_keywords, keywords_all):
    # print(p_ks)
    # print(y_ks)
    # print()

    p_n += len(p_ks)
    g_n += len(y_ks)
    for k in p_ks:
        if k in y_ks:
            correct+=1

#2381 5314 13843
print(correct, p_n, g_n)
precision = correct/p_n
recall = correct/g_n
print('span level f1:', 2*precision*recall/(precision+recall))
print('span level precision:', precision)
print('span level recall:', recall)




