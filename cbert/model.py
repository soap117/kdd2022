import torch
from transformers import get_linear_schedule_with_warmup
import numpy as np
from importlib import import_module
from tqdm import tqdm
from utils.dataset import build_dataset, build_iterator
from modeling_cbert import BertForTokenClassification
from sklearn.metrics import f1_score, accuracy_score
import shutil

PAD, CLS = '[PAD]', '[CLS]'
SEP = '[SEP]'


def build(hidden_size, batch_size, cuda):
    bidirectional = False

    x = import_module('config')
    bert_model = 'hfl/chinese-bert-wwm-ext'
    config = x.Config(batch_size, bert_model)
    train_data = build_dataset(config, './data/train/src_ids.pkl', './data/train/src_masks.pkl',
                               './data/train/tar_masks.pkl')
    test_data = build_dataset(config, './data/test/src_ids.pkl', './data/test/src_masks.pkl',
                              './data/test/tar_masks.pkl')
    val_data = build_dataset(config, './data/valid/src_ids.pkl', './data/valid/src_masks.pkl',
                             './data/valid/tar_masks.pkl')
    train_dataloader = build_iterator(train_data, config)
    val_dataloader = build_iterator(val_data, config)
    test_dataloader = build_iterator(test_data, config)


    model = BertForTokenClassification.from_pretrained(bert_model, num_labels=5)

    if cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    t_total = int(len(train_data) / config.batch_size * config.num_epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(config.warmup_proportion * t_total),
                                                num_training_steps=t_total)

    loss_fun = torch.nn.NLLLoss(reduce=False)
    return model, optimizer, scheduler, train_dataloader, val_dataloader, test_dataloader, loss_fun, config

def valid(model, dataloader, config):
    # validation steps
    tag_values = [0, 1, 2, 3 , 4]
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    predictions, true_labels = [], []
    with open('result/label_test.txt', 'w', encoding='utf-8') as f:
        for i, (batch_src, batch_tar) in tqdm(enumerate(dataloader)):
            x_ids = batch_src[0]
            x_mask = batch_src[1]
            x_indicator = batch_src[2]
            labels = batch_tar
            with torch.no_grad():
                outputs = model(x_ids, attention_mask=x_mask, labels=labels, existing_indicates=x_indicator)
            logits = outputs.logits
            label_ids = labels.to('cpu').numpy()
            loss = outputs.loss
            eval_loss += loss.item()
            predictions.extend([list(p) for p in np.argmax(logits.detach().cpu().numpy(), axis=2)])
            true_labels.extend(label_ids)
            x_ids = x_ids.to('cpu').numpy()
            for x_ids_sample, labels_sample, outputs_sample in zip(x_ids, label_ids, [list(p) for p in np.argmax(logits.detach().cpu().numpy(), axis=2)]):
                tokens = config.tokenizer.convert_ids_to_tokens(x_ids_sample)
                for token, label, output in zip(tokens,labels_sample,outputs_sample):
                    f.write(token+' '+str(label)+' '+str(output)+'\n')
                f.write('\n')

            if i % 100 == 0:
                print('eval loss:%f' % loss.item())
    eval_loss /= len(dataloader)
    pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                 for p_i, l_i in zip(p, l) if tag_values[l_i] != 2]

    valid_tags = [tag_values[l_i] for l in true_labels
                  for l_i in l if tag_values[l_i] != 2]
    acc = accuracy_score(pred_tags, valid_tags)
    f1 = f1_score(pred_tags, valid_tags, average='micro')
    return acc, f1, eval_loss

def train(model, optimizer, scheduler, train_dataloader, val_dataloader, test_dataloader, loss_fun, config):
    # training steps
    max_acc = -99999
    save_file = {}
    for e in range(config.num_epochs):
        model.train()
        train_loss = 0
        for i, (batch_src, batch_tar) in tqdm(enumerate(train_dataloader)):
            # words_1 = ''.join(config.tokenizer.convert_ids_to_tokens(batch_src[0][0]))
            # words_2 = ''.join(config.tokenizer.convert_ids_to_tokens(batch_tar[0][0]))
            x_ids = batch_src[0]
            x_mask = batch_src[1]
            x_indicator = batch_src[2]
            labels = batch_tar
            outputs = model(x_ids, attention_mask=x_mask, labels=labels, existing_indicates=x_indicator)
            loss = outputs.loss
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if i % 100 == 0:
                print('train loss:%f' %loss.item())
        train_loss /= len(train_dataloader)
        print("Train loss: {}".format(train_loss))

        acc, f1, val_loss = valid(model, val_dataloader,config)
        print("Valid loss: {}".format(val_loss))
        print("Validation Accuracy: {}".format(acc))
        print("Validation F1-Score: {}".format(f1))
        if acc > max_acc:
            max_acc = acc
            save_file['epoch'] = e + 1
            save_file['para'] = model.state_dict()
            save_file['best_acc'] = acc
            torch.save(save_file, './cache/best_save.data')
            shutil.copy('result/label_test.txt', 'result/label_test_best.txt')

        print(save_file['epoch'] - 1)


    save_file_best = torch.load('./cache/best_save.data')
    print('Train finished')
    print('Best Val acc:%f' %(save_file_best['best_acc']))
    model.load_state_dict(save_file_best['para'])
    acc, f1, test_loss = valid(model, test_dataloader,config)
    print("Test loss: {}".format(test_loss))
    print("Test Accuracy: {}".format(acc))
    print("Test F1-Score: {}".format(f1))


def main():
    model, optimizer,scheduler, train_dataloader, val_dataloader, test_dataloader, loss_fun, config = build(768, 8, True)
    train(model, optimizer, scheduler, train_dataloader, val_dataloader, test_dataloader, loss_fun, config)
    print('finish')


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()