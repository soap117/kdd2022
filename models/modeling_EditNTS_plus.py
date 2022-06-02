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
MAX_LEN = 512
STOP_ID = tokenizer.vocab['[SEP]']
PAD_ID = tokenizer.vocab['[PAD]']
class EditDecoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, n_layers=1, embedding=None):
        super(EditDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.n_layers = n_layers

        if embedding is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.embedding = embedding
        self.rnn_edits = nn.LSTM(embedding_dim, hidden_size, num_layers=n_layers, batch_first=True)
        self.rnn_words = nn.LSTM(embedding_dim, hidden_size, num_layers=n_layers, batch_first=True)
        self.attn_Projection_org = nn.Linear(hidden_size, hidden_size, bias=False)
        self.initial_hidden = nn.Linear(embedding_dim, 2*hidden_size)
        self.output_hidden_alignment = nn.Linear(embedding_dim, hidden_size, bias=False)
        # self.attn_Projection_scpn = nn.Linear(hidden_size, hidden_size, bias=False) #hard attention here


        self.attn_MLP = nn.Sequential(nn.Linear(hidden_size * 4, embedding_dim),
                                          nn.Tanh())
        self.out = nn.Linear(embedding_dim, self.vocab_size)
        self.out.weight.data = self.embedding.weight.data[:self.vocab_size]

    def execute(self, symbol, input, lm_state):
        """
        :param symbol: token_id for predicted edit action (in teacher forcing mode, give the true one)
        :param input: the word_id being editted currently
        :param lm_state: last lstm state
        :return:
        """
        # predicted_symbol = KEEP -> feed input to RNN_LM
        # predicted_symbol = DEL -> do nothing, return current lm_state
        # predicted_symbol = new word -> feed that word to RNN_LM
        is_keep = torch.eq(symbol, KEEP_ID)
        is_del = torch.eq(symbol, DEL_ID)
        if is_del:
            return lm_state
        elif is_keep: # return lstm with kept word learned in lstm
            _, new_lm_state = self.rnn_words(self.embedding(input.view(-1, 1)), lm_state)
        else: #consider as insert here
            # print(symbol.item())
            input = self.embedding(symbol.view(-1,1))
            _, new_lm_state = self.rnn_words(input,lm_state)
        return new_lm_state

    def execute_batch(self, batch_symbol, batch_input, batch_lm_state):
        batch_h = batch_lm_state[0]
        batch_c = batch_lm_state[1]

        bsz = batch_symbol.size(0)
        unbind_new_h = []
        unbind_new_c = []

        # unbind all batch inputs
        unbind_symbol = torch.unbind(batch_symbol,dim=0)
        unbind_input = torch.unbind(batch_input,dim=0)
        unbind_h = torch.unbind(batch_h,dim=1)
        unbind_c = torch.unbind(batch_c,dim=1)
        for i in range(bsz):
            elem=self.execute(unbind_symbol[i], unbind_input[i], (unbind_h[i].view(1,1,-1), unbind_c[i].view(1,1,-1)))
            unbind_new_h.append(elem[0])
            unbind_new_c.append(elem[1])
        new_batch_lm_h = torch.cat(unbind_new_h,dim=1)
        new_batch_lm_c = torch.cat(unbind_new_c,dim=1)
        return (new_batch_lm_h,new_batch_lm_c)


    def forward(self, input_edits, hidden_org, encoder_outputs_org, org_ids, simp_sent, teacher_forcing_ratio=1.):
        #input_edits: desired output
        #hidden_org initial state
        #simp_sent: desired output with out special marking
        #input_edits and simp_sent need to be padded with START
        bsz, nsteps = input_edits.size()

        # revisit each word and then decide the action, for each action, do the modification and calculate rouge difference
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        decoder_out = []
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

            embedded_words = self.embedding(simp_sent)
            output_words, hidden_words = self.rnn_words(embedded_words, hidden_org)


            key_org = self.attn_Projection_org(output_edits)  # bsz x nsteps x nhid MIGHT USE WORD HERE
            logits_org = torch.bmm(key_org, encoder_outputs_org.transpose(1, 2))  # bsz x nsteps x encsteps
            attn_weights_org = F.softmax(logits_org, dim=-1)  # bsz x nsteps x encsteps
            attn_applied_org = torch.bmm(attn_weights_org, encoder_outputs_org)  # bsz x nsteps x nhid
            #print(org_ids[-1])
            for t in range(nsteps-1):
                # print(t)
                decoder_output_t = output_edits[:, t:t + 1, :]
                attn_applied_org_t = attn_applied_org[:, t:t + 1, :]

                ## find current word
                inds = torch.LongTensor(counter_for_annos)
                ref_word_last = org_ids[-1, counter_for_annos[-1]]
                #print('Current Refer Word:')
                #print(ref_word_last.item())
                dummy = inds.view(-1, 1, 1)
                dummy = dummy.expand(dummy.size(0), dummy.size(1), encoder_outputs_org.size(2)).cuda()
                c = encoder_outputs_org.gather(1, dummy)

                inds = torch.LongTensor(counter_for_keep_ins)
                dummy = inds.view(-1, 1, 1)
                dummy = dummy.expand(dummy.size(0), dummy.size(1), output_words.size(2)).cuda()
                c_word = output_words.gather(1, dummy)
                ref_word_last = simp_sent[-1, counter_for_keep_ins[-1]]
                print('Current Refer Word:')
                print(ref_word_last.item())
                output_t = torch.cat((decoder_output_t, attn_applied_org_t, c,c_word),
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
                counter_for_annos = [i[0] + 1 if i[1] != DEL_ID and i[1] != STOP_ID and i[1] != PAD_ID and i[1] != KEEP_ID and i[2][i[0]+1] == 103 else max(copy.copy(i[3]), i[0])
                                        for i in zip(counter_for_annos, gold_action, org_ids, counter_for_keep_del)]
                print('Current Action:')
                print(gold_action[-1])
                if gold_action[-1] == 1:
                    temp.append(org_ids[-1, r].item())
                    r += 1
                elif gold_action[-1] != DEL_ID and gold_action[-1] != STOP_ID and gold_action[-1] != PAD_ID:
                    temp.append(gold_action[-1].item())
                elif gold_action[-1] == DEL_ID:
                    r += 1
                if temp[-1] != simp_sent[0, len(temp)]:
                    print('here')



                check1 = sum([x >= org_ids.size(1) for x in counter_for_keep_del])
                check2 = sum([x >= simp_sent.size(1) for x in counter_for_keep_ins])
                if check1 or check2:
                    print(org_ids.size(1))
                    print(counter_for_keep_del)
                    break


        else: # no teacher forcing
            decoder_input_edit = input_edits[:, :1]
            decoder_input_word=simp_sent[:,:1]
            t, tt = 0, max(MAX_LEN,input_edits.size(1)-1)

            # initialize
            embedded_edits = self.embedding(decoder_input_edit)
            output_edits, hidden_edits = self.rnn_edits(embedded_edits, hidden_org)

            embedded_words = self.embedding(decoder_input_word)
            output_words, hidden_words = self.rnn_words(embedded_words, hidden_org)
            #
            # # give previous word from tgt simp_sent
            # inds = torch.LongTensor(counter_for_keep_ins)
            # dummy = inds.view(-1, 1, 1)
            # dummy = dummy.expand(dummy.size(0), dummy.size(1), output_words.size(2)).cuda()
            # c_word = output_words.gather(1, dummy)

            while t < tt:
                if t>0:
                    embedded_edits = self.embedding(decoder_input_edit)
                    output_edits, hidden_edits = self.rnn_edits(embedded_edits, hidden_edits)

                key_org = self.attn_Projection_org(output_edits)  # bsz x nsteps x nhid
                logits_org = torch.bmm(key_org, encoder_outputs_org.transpose(1, 2))  # bsz x nsteps x encsteps
                attn_weights_org_t = F.softmax(logits_org, dim=-1)  # bsz x nsteps x encsteps
                attn_applied_org_t = torch.bmm(attn_weights_org_t, encoder_outputs_org)  # bsz x nsteps x nhid

                ## find current word
                inds = torch.LongTensor(counter_for_annos)
                dummy = inds.view(-1, 1, 1)
                dummy = dummy.expand(dummy.size(0), dummy.size(1), encoder_outputs_org.size(2)).cuda()
                c = encoder_outputs_org.gather(1, dummy)

                output_t = torch.cat((output_edits, attn_applied_org_t, c, hidden_words[0].permute(1,0,2)),
                                     2)  # bsz*nsteps x nhid*2
                output_t = self.attn_MLP(output_t)
                output_t = F.log_softmax(self.out(output_t), dim=-1)

                decoder_out.append(output_t)
                decoder_input_edit=torch.argmax(output_t,dim=2)



                # gold_action = input[:, t + 1].vocab_data.cpu().numpy()  # might need to realign here because start added
                pred_action= torch.argmax(output_t,dim=2)
                counter_for_keep_del = [i[0] + 1 if i[1] == 2 or i[1] == 3 or i[1] == 5 else i[0]
                                        for i in zip(counter_for_keep_del, pred_action)]
                counter_for_annos = [
                    i[0] + 1 if i[1] != DEL_ID and i[1] != STOP_ID and i[1] != PAD_ID and i[1] != KEEP_ID and i[2][
                        i[0] + 1] == 103 else max(i[3], i[0])
                    for i in zip(counter_for_annos, pred_action, org_ids, counter_for_keep_del)]

                # update rnn_words
                # find previous generated word
                # give previous word from tgt simp_sent
                dummy_2 = inds.view(-1, 1).cuda()
                org_t = org_ids.gather(1, dummy_2)
                hidden_words = self.execute_batch(pred_action, org_t, hidden_words)  # we give the editted subsequence
                # hidden_words = self.execute_batch(pred_action, org_t, hidden_org)  #here we only give the word

                t += 1
                check = sum([x >= org_ids.size(1) for x in counter_for_keep_del])
                if check:
                    break
        return torch.cat(decoder_out, dim=1), hidden_edits

    def initHidden(self, hidden_encoder_cls):
        h_c = nn.functional.tanh(self.initial_hidden(hidden_encoder_cls))
        h = h_c[:, 0:self.hidden_size]
        c = h_c[:, self.hidden_size:]
        return h.unsqueeze(0).expand(self.n_layers, h.size(0), h.size(1)).contiguous(), c.unsqueeze(0).expand(self.n_layers, h.size(0), h.size(1)).contiguous()


class EditPlus(nn.Module):
    def __init__(self, encoder, decoder, tokenizer):
        super(EditPlus, self).__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.decoder = decoder
        self.hidden_annotation_alignment = nn.Linear(encoder.config.d_model, encoder.config.d_model, bias=False)

    def forward(self, input_ids, decoder_input_ids, anno_position, hidden_annotation, input_edits, org_ids, force_ratio=1.0):
        hidden_annotation = self.hidden_annotation_alignment(hidden_annotation)
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            anno_position=anno_position,
            hidden_annotations=hidden_annotation,
        )
        h_0, c_0 = self.decoder.initHidden(encoder_outputs[0][:, 0])
        decoder_outputs = self.decoder(
            input_edits=input_edits, hidden_org=(h_0,c_0), encoder_outputs_org=encoder_outputs[0][:, 1:], org_ids=input_ids[:, 1:],
            simp_sent=decoder_input_ids, teacher_forcing_ratio = force_ratio
        )


        return decoder_outputs

