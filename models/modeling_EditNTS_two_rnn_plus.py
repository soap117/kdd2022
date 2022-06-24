import torch.nn as nn
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
import torch
import random
import torch.nn.functional as F
import numpy as np
import copy
from transformers import BertTokenizer
from models.modeling_bert_ex import BertModel
bert_model = 'hfl/chinese-bert-wwm-ext'
tokenizer = BertTokenizer.from_pretrained(bert_model)
KEEP_ID = tokenizer.vocab['[unused1]']
DEL_ID = tokenizer.vocab['[unused2]']
INSERT_ID = tokenizer.vocab['[unused5]']
MAX_LEN = 768
STOP_ID = tokenizer.vocab['[SEP]']
PAD_ID = tokenizer.vocab['[PAD]']
class EditDecoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, n_layers=1, embedding=None, output_dim=None):
        super(EditDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        if output_dim is None:
            output_dim = embedding_dim
        if embedding is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.embedding = embedding
        self.rnn_edits = nn.LSTM(embedding_dim, hidden_size, num_layers=n_layers, batch_first=True)
        self.rnn_actions = nn.LSTM(embedding_dim, hidden_size, num_layers=n_layers, batch_first=True)
        self.rnn_words = nn.LSTM(embedding_dim, hidden_size, num_layers=n_layers, batch_first=True)
        self.attn_Projection_org = nn.Linear(hidden_size, hidden_size, bias=False)
        self.initial_hidden = nn.Linear(embedding_dim, 2*hidden_size)
        self.output_hidden_alignment = nn.Linear(embedding_dim, hidden_size, bias=False)
        # self.attn_Projection_scpn = nn.Linear(hidden_size, hidden_size, bias=False) #hard attention here


        self.attn_MLP = nn.Sequential(nn.Linear(hidden_size * 5, output_dim),
                                          nn.Tanh())
        self.attn_ACTION = nn.Sequential(nn.Linear(hidden_size * 5, output_dim),
                                      nn.Tanh())
        self.attn_REF = nn.Sequential(nn.Linear(hidden_size, 2),
                                      nn.Softmax(dim=-1))
        self.out = nn.Linear(output_dim, self.vocab_size)
        self.out_action = nn.Linear(output_dim, 200)
        self.action_mask = torch.ones([1, 200], dtype=torch.float64)
        self.action_mask[0, [0, 1, 2, 5, 101, 102]] = 0
        self.action_mask = self.action_mask * -1e5
        self.action_mask = torch.nn.Parameter(self.action_mask)
        self.out.weight.data = self.embedding.weight.data[:self.vocab_size]
        self.out_action.weight.data = self.embedding.weight.data[:200]

    def execute(self, action, symbol, input, lm_state):
        """
        :param symbol: token_id for predicted edit action (in teacher forcing mode, give the true one)
        :param input: the word_id being editted currently
        :param lm_state: last lstm state
        :return:
        """
        # predicted_symbol = KEEP -> feed input to RNN_LM
        # predicted_symbol = DEL -> do nothing, return current lm_state
        # predicted_symbol = new word -> feed that word to RNN_LM
        is_keep = torch.eq(action, KEEP_ID)
        is_del = torch.eq(action, DEL_ID)
        is_insert = torch.eq(action, INSERT_ID)
        if is_del:
            return lm_state
        elif is_keep: # return lstm with kept word learned in lstm
            _, new_lm_state = self.rnn_words(self.embedding(input.view(-1, 1)), lm_state)
        else: #consider as insert here
            # print(symbol.item())
            input = self.embedding(symbol.view(-1,1))
            _, new_lm_state = self.rnn_words(input, lm_state)
        return new_lm_state

    def execute_batch(self, batch_action, batch_symbol, batch_input, batch_lm_state):
        batch_h = batch_lm_state[0]
        batch_c = batch_lm_state[1]

        bsz = batch_action.size(0)
        unbind_new_h = []
        unbind_new_c = []

        # unbind all batch inputs
        unbind_symbol = torch.unbind(batch_symbol,dim=0)
        unbind_action = torch.unbind(batch_action, dim=0)
        unbind_input = torch.unbind(batch_input,dim=0)
        unbind_h = torch.unbind(batch_h,dim=1)
        unbind_c = torch.unbind(batch_c,dim=1)
        for i in range(bsz):
            elem = self.execute(unbind_action[i], unbind_symbol[i], unbind_input[i], (unbind_h[i].unsqueeze(1).contiguous(), unbind_c[i].unsqueeze(1).contiguous()))
            unbind_new_h.append(elem[0])
            unbind_new_c.append(elem[1])
        new_batch_lm_h = torch.cat(unbind_new_h,dim=1)
        new_batch_lm_c = torch.cat(unbind_new_c,dim=1)
        return (new_batch_lm_h,new_batch_lm_c)


    def forward(self, input_edits, input_actions, hidden_org, encoder_outputs_org, org_ids, simp_sent, teacher_forcing_ratio=1., eval=False, clean_indication=None):
        #input_edits: desired output
        #hidden_org initial state
        #simp_sent: desired output with out special marking
        #input_edits and simp_sent need to be padded with START
        bsz, nsteps = input_edits.size()

        # revisit each word and then decide the action, for each action, do the modification and calculate rouge difference
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        decoder_out = []
        decoder_out_action = []
        counter_for_keep_del = np.zeros(bsz, dtype=int)
        counter_for_keep_ins =np.zeros(bsz, dtype=int)
        counter_for_annos = np.zeros(bsz, dtype=int)
        encoder_outputs_org = F.tanh(self.output_hidden_alignment(encoder_outputs_org))
        # decoder in the training:

        if use_teacher_forcing:
            r = 0
            temp = []
            embedded_edits = self.embedding(input_edits)
            output_edits, hidden_edits = self.rnn_edits(embedded_edits, hidden_org)

            embedded_actions = self.embedding(input_actions)
            output_actions, hidden_actions = self.rnn_actions(embedded_actions, hidden_org)

            embedded_words = self.embedding(simp_sent)
            output_words, hidden_words = self.rnn_words(embedded_words, hidden_org)


            #key_org = self.attn_Projection_org(output_actions)  # bsz x nsteps x nhid MIGHT USE WORD HERE
            #logits_org = torch.bmm(key_org, encoder_outputs_org.transpose(1, 2))  # bsz x nsteps x encsteps
            #attn_weights_org = F.softmax(logits_org, dim=-1)  # bsz x nsteps x encsteps
            #attn_applied_org = torch.bmm(attn_weights_org, encoder_outputs_org)  # bsz x nsteps x nhid
            #print(org_ids[-1])
            for t in range(nsteps-1):
                # print(t)
                decoder_output_t = output_edits[:, t:t + 1, :]
                decoder_action_t = output_actions[:, t:t + 1, :]

                ## find current annotation word
                inds = torch.LongTensor(counter_for_annos)
                #ref_word_last = org_ids[-1, counter_for_annos[-1]]
                #print('Current Refer Word:')
                #print(ref_word_last.item())
                dummy = inds.view(-1, 1, 1)
                dummy = dummy.expand(dummy.size(0), dummy.size(1), encoder_outputs_org.size(2)).cuda()
                c_anno = encoder_outputs_org.gather(1, dummy)
                ## find current input word
                inds = torch.LongTensor(counter_for_keep_del)
                #ref_word_last = org_ids[-1, counter_for_annos[-1]]
                # print('Current Refer Word:')
                # print(ref_word_last.item())
                dummy = inds.view(-1, 1, 1)
                dummy = dummy.expand(dummy.size(0), dummy.size(1), encoder_outputs_org.size(2)).cuda()
                c_input = encoder_outputs_org.gather(1, dummy)

                c = torch.cat([c_input, c_anno], dim=1)
                weight_ref = self.attn_REF(decoder_output_t)
                c_edit = torch.bmm(weight_ref, c)
                weight_ref = self.attn_REF(decoder_action_t)
                c_action = torch.bmm(weight_ref, c)


                inds = torch.LongTensor(counter_for_keep_ins)
                dummy = inds.view(-1, 1, 1)
                dummy = dummy.expand(dummy.size(0), dummy.size(1), output_words.size(2)).cuda()
                c_word = output_words.gather(1, dummy)

                key_org = self.attn_Projection_org(c_word)  # bsz x nsteps x nhid
                logits_org = torch.bmm(key_org, encoder_outputs_org.transpose(1, 2))  # bsz x nsteps x encsteps
                attn_weights_org_t = F.softmax(logits_org, dim=-1)  # bsz x nsteps x encsteps
                attn_applied_org_t = torch.bmm(attn_weights_org_t, encoder_outputs_org)  # bsz x nsteps x nhid

                output_action = torch.cat((decoder_action_t, decoder_output_t, attn_applied_org_t, c_action, c_word),
                                          2)  # bsz*nsteps x nhid*2
                output_action = self.attn_ACTION(output_action)
                output_action = F.log_softmax(self.out_action(output_action) + self.action_mask, dim=-1)
                decoder_out_action.append(output_action)

                output_t = torch.cat((decoder_output_t, decoder_action_t, attn_applied_org_t, c_edit, c_word),
                                     2)  # bsz*nsteps x nhid*2
                output_t = self.attn_MLP(output_t)
                output_t = F.log_softmax(self.out(output_t), dim=-1)
                decoder_out.append(output_t)




                # interpreter's output from lm
                gold_action = input_edits[:, t + 1].data.cpu().numpy()  # might need to realign here because start added
                counter_for_keep_del = [i[0] + 1 if i[1] == KEEP_ID or i[1] == DEL_ID  else i[0]
                                        for i in zip(counter_for_keep_del, gold_action)]
                counter_for_keep_ins = [i[0] + 1 if i[1] != DEL_ID and i[1] != STOP_ID and i[1] != PAD_ID else i[0]
                                        for i in zip(counter_for_keep_ins, gold_action)]
                counter_for_annos = [i[0] + 1 if i[1] != DEL_ID and i[1] != STOP_ID and i[1] != PAD_ID and i[1] != KEEP_ID and i[0]+1 < len(i[2]) and i[2][i[0]+1] in [103, 3, 4] else max(copy.copy(i[3]), i[0])
                                        for i in zip(counter_for_annos, gold_action, org_ids, counter_for_keep_del)]
                #print('Current Action:')
                #print(gold_action[-1])
                if gold_action[-1] == 1:
                    temp.append(org_ids[-1, r].item())
                    r += 1
                elif gold_action[-1] != DEL_ID and gold_action[-1] != STOP_ID and gold_action[-1] != PAD_ID:
                    temp.append(gold_action[-1].item())
                elif gold_action[-1] == DEL_ID:
                    r += 1



                check1 = sum([x >= org_ids.size(1) for x in counter_for_keep_del])
                check2 = sum([x >= simp_sent.size(1) for x in counter_for_keep_ins])
                if check1:
                    #print('run out input')
                    #print(org_ids.size(1))
                    #print(counter_for_keep_del)
                    break
                if check2:
                    #print('run out target')
                    #print(simp_sent.size(1))
                    #print(counter_for_keep_ins)
                    break


        else: # no teacher forcing
            decoder_input_edit = input_edits[:, :1]
            decoder_input_action = input_actions[:, :1]
            decoder_input_word = simp_sent[:,:1]
            t, tt = 0, max(MAX_LEN,input_edits.size(1)-1)

            # initialize
            embedded_edits = self.embedding(decoder_input_edit)
            output_edits_h, hidden_edits = self.rnn_edits(embedded_edits, hidden_org)

            embedded_actions = self.embedding(decoder_input_action)
            output_actions_h, hidden_actions = self.rnn_edits(embedded_actions, hidden_org)

            embedded_words = self.embedding(decoder_input_word)
            output_words_h, hidden_words = self.rnn_words(embedded_words, hidden_org)
            #
            # # give previous word from tgt simp_sent
            # inds = torch.LongTensor(counter_for_keep_ins)
            # dummy = inds.view(-1, 1, 1)
            # dummy = dummy.expand(dummy.size(0), dummy.size(1), output_words.size(2)).cuda()
            # c_word = output_words.gather(1, dummy)
            inserts = 0
            while t < tt:
                if t>0:
                    embedded_edits = self.embedding(decoder_input_edit)
                    output_edits_h, hidden_edits = self.rnn_edits(embedded_edits, hidden_edits)

                    embedded_actions = self.embedding(decoder_input_action)
                    output_actions_h, hidden_actions = self.rnn_actions(embedded_actions, hidden_actions)

                key_org = self.attn_Projection_org(output_words_h)  # bsz x nsteps x nhid
                logits_org = torch.bmm(key_org, encoder_outputs_org.transpose(1, 2))  # bsz x nsteps x encsteps
                attn_weights_org_t = F.softmax(logits_org, dim=-1)  # bsz x nsteps x encsteps
                attn_applied_org_t = torch.bmm(attn_weights_org_t, encoder_outputs_org)  # bsz x nsteps x nhid

                ## find current word
                inds = torch.LongTensor(counter_for_annos)
                dummy = inds.view(-1, 1, 1)
                dummy = dummy.expand(dummy.size(0), dummy.size(1), encoder_outputs_org.size(2)).cuda()
                c_anno = encoder_outputs_org.gather(1, dummy)

                inds = torch.LongTensor(counter_for_keep_del)
                dummy = inds.view(-1, 1, 1)
                dummy = dummy.expand(dummy.size(0), dummy.size(1), encoder_outputs_org.size(2)).cuda()
                c_input = encoder_outputs_org.gather(1, dummy)
                if eval:
                    c_inds = org_ids.gather(1, inds.view(-1, 1).cuda())

                c = torch.cat([c_input, c_anno], dim=1)
                weight_ref = self.attn_REF(output_edits_h)
                c_edit = torch.bmm(weight_ref, c)
                weight_ref = self.attn_REF(output_actions_h)
                c_action = torch.bmm(weight_ref, c)

                output_action = torch.cat((output_actions_h, output_edits_h, attn_applied_org_t, c_action, hidden_words[0][-1].unsqueeze(1)),
                                     2)  # bsz*nsteps x nhid*2
                output_action = self.attn_ACTION(output_action)
                output_action = F.log_softmax(self.out_action(output_action) + self.action_mask, dim=-1)

                output_edit = torch.cat((output_edits_h, output_actions_h, attn_applied_org_t, c_edit, hidden_words[0][-1].unsqueeze(1)),
                                     2)  # bsz*nsteps x nhid*2
                output_edit = self.attn_MLP(output_edit)
                output_edit = F.log_softmax(self.out(output_edit), dim=-1)
                if eval:
                    if c_inds == 8020 or c_inds == 8021 or c_inds==109:
                        output_action[:,:, 1] += 1e10
                    if clean_indication is not None:
                        clean_inds = clean_indication.gather(1, inds.view(-1, 1).cuda())
                        if clean_inds == 0:
                            output_action[:, :, 1] += 1e10
                    pred_action = torch.argmax(output_action, dim=2)
                    if pred_action == 5:
                        pred_word = torch.argmax(output_edit, dim=2)
                        inserts += 1
                        if inserts > 20:
                            inserts = 0
                            output_action[:, :, 2] += 1e10
                        if inserts == 1:
                            old_pred_word = -1
                        if pred_word == old_pred_word:
                            inserts = 0
                            output_action[:, :, 2] += 1e10
                        old_pred_word = pred_word
                decoder_out.append(output_edit)
                decoder_out_action.append(output_action)
                decoder_input_action = torch.argmax(output_action, dim=2)
                decoder_input_edit = torch.argmax(output_edit, dim=2)
                decoder_input_edit = torch.where(decoder_input_action != INSERT_ID, decoder_input_action, decoder_input_edit)



                # gold_action = input[:, t + 1].vocab_data.cpu().numpy()  # might need to realign here because start added
                pred_action = torch.argmax(output_action, dim=2)
                pred_edit = torch.argmax(output_edit, dim=2)
                pred_edit = torch.where(pred_action != INSERT_ID,  pred_action,
                                                 pred_edit)
                inds = torch.LongTensor(counter_for_keep_del)
                counter_for_keep_del = [i[0] + 1 if i[1] == KEEP_ID or i[1] == DEL_ID else i[0]
                                        for i in zip(counter_for_keep_del, pred_action)]
                counter_for_annos = [
                    i[0] + 1 if i[1] != DEL_ID and i[1] != STOP_ID and i[1] != PAD_ID and i[1] != KEEP_ID and i[
                        0] + 1 < len(i[2]) and i[2][i[0] + 1] in [103, 3, 4] else max(copy.copy(i[3]), i[0])
                    for i in zip(counter_for_annos, pred_action, org_ids, counter_for_keep_del)]

                # update rnn_words
                # find previous generated word
                # give previous word from tgt simp_sent
                dummy_2 = inds.view(-1, 1).cuda()
                org_t = org_ids.gather(1, dummy_2)
                hidden_words = self.execute_batch(pred_action, pred_edit, org_t, hidden_words)  # we give the editted subsequence
                output_words_h = hidden_words[0][0:1].permute(1,0,2)
                # hidden_words = self.execute_batch(pred_action, org_t, hidden_org)  #here we only give the word

                t += 1
                check = sum([x >= org_ids.size(1) for x in counter_for_keep_del])
                check2 = sum([x >= org_ids.size(1) for x in counter_for_annos])
                if check or check2:
                    break
        return torch.cat(decoder_out_action, dim=1), torch.cat(decoder_out, dim=1), hidden_edits

    def initHidden(self, hidden_encoder_cls):
        h_c = nn.functional.tanh(self.initial_hidden(hidden_encoder_cls))
        h = h_c[:, 0:self.hidden_size]
        c = h_c[:, self.hidden_size:]
        return h.unsqueeze(0).expand(self.n_layers, h.size(0), h.size(1)).contiguous(), c.unsqueeze(0).expand(self.n_layers, h.size(0), h.size(1)).contiguous()

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
def unsort(x_sorted, sorted_order):
    x_unsort = torch.zeros_like(x_sorted)
    x_unsort[:, sorted_order,:] = x_sorted
    return x_unsort
class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pos_vocab_size, pos_embedding_dim,hidden_size, n_layers=1, embedding=None, embeddingPOS=None,dropout=0.3):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        if embedding is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.embedding = embedding

        if embeddingPOS is None:
            self.embeddingPOS = nn.Embedding(pos_vocab_size, pos_embedding_dim)
        else:
            self.embeddingPOS = embeddingPOS

        self.rnn = nn.LSTM(embedding_dim+pos_embedding_dim, hidden_size, num_layers=n_layers, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, input_ids, anno_position, hidden_annotation, hidden):
        #inp and inp pose should be both sorted
        inp_sorted=inp[0]
        inp_lengths_sorted=inp[1].cpu()
        inp_sort_order=inp[2]

        inp_pos_sorted = inp_pos[0]
        inp_pos_lengths_sorted = inp_pos[1]
        inp_pos_sort_order = inp_pos[2]

        emb = self.embedding(inp_sorted)
        emb_pos = self.embeddingPOS(inp_pos_sorted)

        embed_cat = torch.cat((emb,emb_pos),dim=2)
        packed_emb = pack(embed_cat, inp_lengths_sorted,batch_first=True)
        memory_bank, encoder_final = self.rnn(packed_emb, hidden)

        memory_bank = unpack(memory_bank)[0]
        memory_bank = unsort(memory_bank, inp_sort_order)

        h_unsorted=unsort(encoder_final[0], inp_sort_order)
        c_unsorted=unsort(encoder_final[1], inp_sort_order)


        return memory_bank.transpose(0,1), (h_unsorted,c_unsorted)

    def initHidden(self, bsz):
        weight = next(self.parameters()).data
        return Variable(weight.new(self.n_layers * 2, bsz, self.hidden_size).zero_()), \
               Variable(weight.new(self.n_layers * 2, bsz, self.hidden_size).zero_())

class EditPlus(nn.Module):
    def __init__(self, encoder, decoder, tokenizer):
        super(EditPlus, self).__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.decoder = decoder
        self.hidden_annotation_alignment = nn.Linear(encoder.config.d_model, encoder.config.d_model, bias=False)

    def forward(self, input_ids, decoder_input_ids, anno_position, hidden_annotation, input_edits, input_actions, org_ids=None, force_ratio=1.0, eval=False, clean_indication=None):
        if hidden_annotation is not None:
            hidden_annotation = self.hidden_annotation_alignment(hidden_annotation)
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            anno_position=anno_position,
            hidden_annotations=hidden_annotation,
        )
        h_0, c_0 = self.decoder.initHidden(encoder_outputs[0][:, 0])
        decoder_outputs = self.decoder(
            input_edits=input_edits, input_actions=input_actions, hidden_org=(h_0,c_0), encoder_outputs_org=encoder_outputs[0][:, 1:], org_ids=input_ids[:, 1:],
            simp_sent=decoder_input_ids, teacher_forcing_ratio=force_ratio, eval=eval, clean_indication=clean_indication
        )

        return decoder_outputs

