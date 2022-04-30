
from transformers import BertTokenizer, BertModel


bert_model = 'hfl/chinese-bert-wwm-ext'
tokenizer = BertTokenizer.from_pretrained(bert_model)
tokenizer.add_tokens(["痛风石"])
tokenizer.add_tokens(["痛风"])



print(len(tokenizer))
print(tokenizer.tokenize('白癜风得到的依从性关于其他的痛风石'))

