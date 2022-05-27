from model import build
import torch
from tqdm import tqdm
import numpy as np

model, optimizer,scheduler, train_dataloader, val_dataloader, test_dataloader, loss_fun, config = build(768, 8, True)

save_file_best = torch.load('./cache/best_save.data')
model.load_state_dict(save_file_best['para'])

tag_values = [0, 1, 2, 3, 4]
model.eval()
eval_loss, eval_accuracy = 0, 0
predictions, true_labels = [], []
for i, (batch_src, batch_tar) in tqdm(enumerate(test_dataloader)):
    x_ids = batch_src[0]
    x_mask = batch_src[1]
    x_indicates = batch_src[2]
    labels = batch_tar
    with torch.no_grad():
        outputs = model(x_ids, existing_indicates=x_indicates, labels=labels)
    logits = outputs.logits
    label_ids = labels.to('cpu').numpy()
    loss = outputs.loss
    eval_loss += loss.item()
    predictions = [list(p) for p in np.argmax(logits.detach().cpu().numpy(), axis=2)]
    true_labels.extend(label_ids)
    for src, labels in zip(batch_src[0],predictions):
        src = config.tokenizer.convert_ids_to_tokens(src)
        l, r = 0, 0
        keywords = []
        while l<len(labels):
            while l<len(labels) and (labels[l] == 0 or labels[l] == 4): l += 1
            r = l+1
            while r<len(labels) and labels[r] != 0 and labels[r] != 4 and labels[r-1] != 3: r += 1
            keyword = src[l:r]
            if l<len(labels) and labels[l] == 1:
                keyword = ''.join(keyword)
                if keyword != '':
                    keywords.append(keyword)
            l = r
        print(''.join(src))
        print(keywords)
