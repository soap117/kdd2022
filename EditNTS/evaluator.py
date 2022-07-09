import numpy as np
import torch
from nltk.translate.bleu_score import *

smooth = SmoothingFunction()
from SARI import SARIsent
import nltk
import data
nltk.data.path.append("/media/nvme/nltk_data")
from label_edits import edit2sent


def sort_by_lens(seq, seq_lengths):
    seq_lengths_sorted, sort_order = seq_lengths.sort(descending=True)
    seq_sorted = seq.index_select(0, sort_order)
    return seq_sorted, seq_lengths_sorted, sort_order

import nltk

def cal_bleu_score(decoded, target):
    return nltk.translate.bleu_score.sentence_bleu([target], decoded,
            smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1)

class Evaluator():
    """"""
    def __init__(self, loss, batch_size=64):
        self.loss = loss
        self.batch_size = batch_size

    def evaluate(self, dataset, vocab, model, args, max_edit_steps=50):
        """ Evaluate a model on given dataset and return performance during training
        Args:
            dataset: an object of mydata.Dataset()
            model (editNTS model): model to evaluate
            vocab: an object containing mydata.Vocab()
            args: args from the main methods
        Returns:
            loss (float): loss of the given model on the given dataset evaluated with teacher forcing
            sari: computed based on python script

        """
        print_loss, print_loss_tf = [], []
        bleu_list = []
        ter = 0.
        sari_list = []
        sys_out=[]

        print('Doing tokenized evaluation')
        for i, batch_df in dataset.batch_generator(batch_size=1, shuffle=False):
            model.eval()
            prepared_batch, syn_tokens_list = data.prepare_batch(batch_df, vocab, args.max_seq_len)  # comp,scpn,simp

            org_ids = prepared_batch[0]
            org_lens = org_ids.ne(0).sum(1)
            org = sort_by_lens(org_ids, org_lens)  # inp=[inp_sorted, inp_lengths_sorted, inp_sort_order]

            org_pos_ids = prepared_batch[1]
            org_pos_lens = org_pos_ids.ne(0).sum(1)
            org_pos = sort_by_lens(org_pos_ids, org_pos_lens)  # inp=[inp_sorted, inp_lengths_sorted, inp_sort_order]

            out = prepared_batch[2][:, :]
            tar = prepared_batch[2][:, 1:]
            simp_ids = prepared_batch[3]

            # best_seq_list = model.beamsearch(org, out,simp_ids, org_ids, org_pos, 5)
            output_without_teacher_forcing = model(org, out, org_ids, org_pos, simp_ids,0.0)
            output_teacher_forcing = model(org, out, org_ids, org_pos,simp_ids, 1.0)

            if True: # the loss on validation is computed based on teacher forcing
                if len(output_without_teacher_forcing) == 1:
                    ##################calculate loss
                    tar_lens = tar.ne(0).sum(1).float()
                    tar_flat = tar.contiguous().view(-1).type(torch.LongTensor).cuda()
                    def compute_loss(output,tar_flat): #this function computes the loss based on model outputs and target in flat
                        loss = self.loss(output.contiguous().view(-1, vocab.count), tar_flat).contiguous()
                        loss[tar_flat == 1] = 0  # remove loss for UNK
                        loss = loss.view(tar.size())
                        loss = loss.sum(1).float()
                        loss = loss / tar_lens
                        loss = loss.mean()
                        return loss
                    loss_tf = compute_loss(output_teacher_forcing,tar_flat)
                    print_loss_tf.append(loss_tf.item())
                else:
                    ##################calculate loss

                    def compute_loss(output,
                                     tar_flat, size_vocb):  # this function computes the loss based on model outputs and target in flat
                        loss = self.loss(output.contiguous().view(-1, size_vocb), tar_flat).contiguous()
                        loss[tar_flat == 1] = 0  # remove loss for UNK
                        loss = loss.view(tar.size())
                        loss = loss.sum(1).float()
                        loss = loss / tar_lens
                        loss = loss.mean()
                        return loss

                    output_action, output_edit = output_teacher_forcing[0], output_teacher_forcing[1]
                    tar_action = torch.zeros_like(tar) + 6
                    tar_action = torch.where(
                        (tar == 0) | (tar == 1) | (tar == 2) | (tar == 3) | (tar == 4) | (tar == 5), tar, tar_action)
                    tar_edit = torch.zeros_like(tar)
                    tar_edit = torch.where((tar == 0) | (tar == 1) | (tar == 2) | (tar == 3) | (tar == 4) | (tar == 5),
                                           tar_edit, tar)
                    tar_lens = tar_action.ne(0).sum(1).float()+1e-5
                    tar_flat = tar_action.contiguous().view(-1).type(torch.LongTensor).cuda()
                    loss_tf_action = compute_loss(output_action, tar_flat, 7)

                    tar_lens = tar_edit.ne(0).sum(1).float()+1e-5
                    tar_flat = tar_edit.contiguous().view(-1).type(torch.LongTensor).cuda()
                    loss_tf_edit = compute_loss(output_edit, tar_flat, vocab.count)

                    loss_tf = loss_tf_action + loss_tf_edit
                    print_loss_tf.append(loss_tf.item())

            # the SARI and BLUE is computed based on model.eval without teacher forcing
                ## write beam search here
                # try:
            if len(output_without_teacher_forcing) == 1:
                for j in range(output_without_teacher_forcing.size()[0]):
                    example = batch_df.iloc[j]
                    example_out = output_without_teacher_forcing[j, :, :]

                    ##GREEDY
                    pred_action = torch.argmax(example_out, dim=1).view(-1).data.cpu().numpy()
                    edit_list_in_tokens = data.id2edits(pred_action, vocab)
                    # ###BEST BEAM
                    # edit_list_in_tokens = vocab_data.id2edits(best_seq_list[0][1:], vocab)

                    greedy_decoded_tokens = ' '.join(edit2sent(example['comp_tokens'], edit_list_in_tokens))
                    greedy_decoded_tokens = greedy_decoded_tokens.split('STOP')[0].split(' ')
                    # tgt_tokens_translated = [vocab.i2w[i] for i in example['simp_ids']]
                    sys_out.append(' '.join(greedy_decoded_tokens))

                    # prt = True if random.random() < 0.01 else False
                    # if prt:
                    #     print('*' * 30)
                    #     # print('tgt_in_tokens_translated', ' '.join(tgt_tokens_translated))
                    #     print('ORG', ' '.join(example['comp_tokens']))
                    #     print('GEN', ' '.join(greedy_decoded_tokens))
                    #     print('TGT', ' '.join(example['simp_tokens']))
                    #     print('edit_list_in_tokens',edit_list_in_tokens)
                    #     print('gold labels', ' '.join(example['edit_labels']))

                    bleu_list.append(cal_bleu_score(greedy_decoded_tokens, example['simp_tokens']))

                    # calculate sari
                    comp_string = ' '.join(example['comp_tokens'])
                    simp_string = ' '.join(example['simp_tokens'])
                    gen_string = ' '.join(greedy_decoded_tokens)
                    sari_list.append(SARIsent(comp_string, gen_string, [simp_string]))
            else:
                for j in range(output_without_teacher_forcing[0].size()[0]):
                    example = batch_df.iloc[j]
                    output_action, output_edit = output_without_teacher_forcing[0], output_without_teacher_forcing[1]
                    example_out_action = output_action[j, :, :]
                    example_out_edit = output_edit[j, :, :]
                    ##GREEDY
                    pred_action = torch.argmax(example_out_action, dim=1).view(-1).data.cpu().numpy()
                    pred_edit = torch.argmax(example_out_edit, dim=1).view(-1).data.cpu().numpy()
                    pred_action = np.where(pred_action != 6, pred_action,
                                            pred_edit)

                    edit_list_in_tokens = data.id2edits(pred_action, vocab)
                    # ###BEST BEAM
                    # edit_list_in_tokens = vocab_data.id2edits(best_seq_list[0][1:], vocab)

                    greedy_decoded_tokens = ' '.join(edit2sent(example['comp_tokens'], edit_list_in_tokens))
                    greedy_decoded_tokens = greedy_decoded_tokens.split('STOP')[0].split(' ')
                    # tgt_tokens_translated = [vocab.i2w[i] for i in example['simp_ids']]
                    sys_out.append(' '.join(greedy_decoded_tokens))

                    # prt = True if random.random() < 0.01 else False
                    # if prt:
                    #     print('*' * 30)
                    #     # print('tgt_in_tokens_translated', ' '.join(tgt_tokens_translated))
                    #     print('ORG', ' '.join(example['comp_tokens']))
                    #     print('GEN', ' '.join(greedy_decoded_tokens))
                    #     print('TGT', ' '.join(example['simp_tokens']))
                    #     print('edit_list_in_tokens',edit_list_in_tokens)
                    #     print('gold labels', ' '.join(example['edit_labels']))

                    bleu_list.append(cal_bleu_score(greedy_decoded_tokens, example['simp_tokens']))

                    # calculate sari
                    comp_string = ' '.join(example['comp_tokens'])
                    simp_string = ' '.join(example['simp_tokens'])
                    gen_string = ' '.join(greedy_decoded_tokens)
                    sari_list.append(SARIsent(comp_string, gen_string, [simp_string]))

        print('loss_with_teacher_forcing', np.mean(print_loss_tf))
        model.train()
        return np.mean(print_loss_tf), np.mean(bleu_list), np.mean(sari_list), sys_out
    def evaluate_ind(self, dataset, vocab, model, args, max_edit_steps=50):
        """ Evaluate a model on given dataset and return performance during training
        Args:
            dataset: an object of mydata.Dataset()
            model (editNTS model): model to evaluate
            vocab: an object containing mydata.Vocab()
            args: args from the main methods
        Returns:
            loss (float): loss of the given model on the given dataset evaluated with teacher forcing
            sari: computed based on python script

        """
        print_loss, print_loss_tf = [], []
        bleu_list = []
        ter = 0.
        sari_list = []
        sys_out=[]

        print('Doing tokenized evaluation')
        for i, batch_df in dataset.batch_generator(batch_size=1, shuffle=False):
            model.eval()
            prepared_batch, syn_tokens_list = data.prepare_batch_indication(batch_df, vocab, args.max_seq_len)  # comp,scpn,simp

            org_ids = prepared_batch[0]
            org_lens = org_ids.ne(0).sum(1)
            org = sort_by_lens(org_ids, org_lens)  # inp=[inp_sorted, inp_lengths_sorted, inp_sort_order]

            org_indication_ids = prepared_batch[1]
            org_indication_lens = org_indication_ids.ne(0).sum(1)
            org_indication = sort_by_lens(org_indication_ids, org_indication_lens)

            org_pos_ids = prepared_batch[2]
            org_pos_lens = org_pos_ids.ne(0).sum(1)
            org_pos = sort_by_lens(org_pos_ids, org_pos_lens)  # inp=[inp_sorted, inp_lengths_sorted, inp_sort_order]

            out = prepared_batch[3][:, :]
            tar = prepared_batch[3][:, 1:]
            simp_ids = prepared_batch[4]

            # best_seq_list = model.beamsearch(org, out,simp_ids, org_ids, org_pos, 5)
            output_without_teacher_forcing = model(org, out, org_ids, org_pos, org_indication, simp_ids,0.0)
            output_teacher_forcing = model(org, out, org_ids, org_pos, org_indication, simp_ids, 1.0)

            if True: # the loss on validation is computed based on teacher forcing
                if len(output_without_teacher_forcing) == 1:
                    ##################calculate loss
                    tar_lens = tar.ne(0).sum(1).float()
                    tar_flat = tar.contiguous().view(-1).type(torch.LongTensor).cuda()
                    def compute_loss(output,tar_flat): #this function computes the loss based on model outputs and target in flat
                        loss = self.loss(output.contiguous().view(-1, vocab.count), tar_flat).contiguous()
                        loss[tar_flat == 1] = 0  # remove loss for UNK
                        loss = loss.view(tar.size())
                        loss = loss.sum(1).float()
                        loss = loss / tar_lens
                        loss = loss.mean()
                        return loss
                    loss_tf = compute_loss(output_teacher_forcing,tar_flat)
                    print_loss_tf.append(loss_tf.item())
                else:
                    ##################calculate loss

                    def compute_loss(output,
                                     tar_flat, size_vocb):  # this function computes the loss based on model outputs and target in flat
                        loss = self.loss(output.contiguous().view(-1, size_vocb), tar_flat).contiguous()
                        loss[tar_flat == 1] = 0  # remove loss for UNK
                        loss = loss.view(tar.size())
                        loss = loss.sum(1).float()
                        loss = loss / tar_lens
                        loss = loss.mean()
                        return loss

                    output_action, output_edit = output_teacher_forcing[0], output_teacher_forcing[1]
                    tar_action = torch.zeros_like(tar) + 6
                    tar_action = torch.where(
                        (tar == 0) | (tar == 1) | (tar == 2) | (tar == 3) | (tar == 4) | (tar == 5), tar, tar_action)
                    tar_edit = torch.zeros_like(tar)
                    tar_edit = torch.where((tar == 0) | (tar == 1) | (tar == 2) | (tar == 3) | (tar == 4) | (tar == 5),
                                           tar_edit, tar)
                    tar_lens = tar_action.ne(0).sum(1).float()+1e-5
                    tar_flat = tar_action.contiguous().view(-1).type(torch.LongTensor).cuda()
                    loss_tf_action = compute_loss(output_action, tar_flat, 7)

                    tar_lens = tar_edit.ne(0).sum(1).float()+1e-5
                    tar_flat = tar_edit.contiguous().view(-1).type(torch.LongTensor).cuda()
                    loss_tf_edit = compute_loss(output_edit, tar_flat, vocab.count)

                    loss_tf = loss_tf_action + loss_tf_edit
                    print_loss_tf.append(loss_tf.item())

            # the SARI and BLUE is computed based on model.eval without teacher forcing
                ## write beam search here
                # try:
            if len(output_without_teacher_forcing) == 1:
                for j in range(output_without_teacher_forcing.size()[0]):
                    example = batch_df.iloc[j]
                    example_out = output_without_teacher_forcing[j, :, :]

                    ##GREEDY
                    pred_action = torch.argmax(example_out, dim=1).view(-1).data.cpu().numpy()
                    edit_list_in_tokens = data.id2edits(pred_action, vocab)
                    # ###BEST BEAM
                    # edit_list_in_tokens = vocab_data.id2edits(best_seq_list[0][1:], vocab)

                    greedy_decoded_tokens = ' '.join(edit2sent(example['comp_tokens'], edit_list_in_tokens))
                    greedy_decoded_tokens = greedy_decoded_tokens.split('STOP')[0].split(' ')
                    # tgt_tokens_translated = [vocab.i2w[i] for i in example['simp_ids']]
                    sys_out.append(' '.join(greedy_decoded_tokens))

                    # prt = True if random.random() < 0.01 else False
                    # if prt:
                    #     print('*' * 30)
                    #     # print('tgt_in_tokens_translated', ' '.join(tgt_tokens_translated))
                    #     print('ORG', ' '.join(example['comp_tokens']))
                    #     print('GEN', ' '.join(greedy_decoded_tokens))
                    #     print('TGT', ' '.join(example['simp_tokens']))
                    #     print('edit_list_in_tokens',edit_list_in_tokens)
                    #     print('gold labels', ' '.join(example['edit_labels']))

                    bleu_list.append(cal_bleu_score(greedy_decoded_tokens, example['simp_tokens']))

                    # calculate sari
                    comp_string = ' '.join(example['comp_tokens'])
                    simp_string = ' '.join(example['simp_tokens'])
                    gen_string = ' '.join(greedy_decoded_tokens)
                    sari_list.append(SARIsent(comp_string, gen_string, [simp_string]))
            else:
                for j in range(output_without_teacher_forcing[0].size()[0]):
                    example = batch_df.iloc[j]
                    output_action, output_edit = output_without_teacher_forcing[0], output_without_teacher_forcing[1]
                    example_out_action = output_action[j, :, :]
                    example_out_edit = output_edit[j, :, :]
                    ##GREEDY
                    pred_action = torch.argmax(example_out_action, dim=1).view(-1).data.cpu().numpy()
                    pred_edit = torch.argmax(example_out_edit, dim=1).view(-1).data.cpu().numpy()
                    pred_action = np.where(pred_action != 6, pred_action,
                                            pred_edit)

                    edit_list_in_tokens = data.id2edits(pred_action, vocab)
                    # ###BEST BEAM
                    # edit_list_in_tokens = vocab_data.id2edits(best_seq_list[0][1:], vocab)

                    greedy_decoded_tokens = ' '.join(edit2sent(example['comp_tokens'], edit_list_in_tokens))
                    greedy_decoded_tokens = greedy_decoded_tokens.split('STOP')[0].split(' ')
                    # tgt_tokens_translated = [vocab.i2w[i] for i in example['simp_ids']]
                    sys_out.append(' '.join(greedy_decoded_tokens))

                    # prt = True if random.random() < 0.01 else False
                    # if prt:
                    #     print('*' * 30)
                    #     # print('tgt_in_tokens_translated', ' '.join(tgt_tokens_translated))
                    #     print('ORG', ' '.join(example['comp_tokens']))
                    #     print('GEN', ' '.join(greedy_decoded_tokens))
                    #     print('TGT', ' '.join(example['simp_tokens']))
                    #     print('edit_list_in_tokens',edit_list_in_tokens)
                    #     print('gold labels', ' '.join(example['edit_labels']))

                    bleu_list.append(cal_bleu_score(greedy_decoded_tokens, example['simp_tokens']))

                    # calculate sari
                    comp_string = ' '.join(example['comp_tokens'])
                    simp_string = ' '.join(example['simp_tokens'])
                    gen_string = ' '.join(greedy_decoded_tokens)
                    sari_list.append(SARIsent(comp_string, gen_string, [simp_string]))

        print('loss_with_teacher_forcing', np.mean(print_loss_tf))
        model.train()
        return np.mean(print_loss_tf), np.mean(bleu_list), np.mean(sari_list), sys_out