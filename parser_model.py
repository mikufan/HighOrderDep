import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import *
from torch import optim
import eisner_layer_m1 as EL_M1
import utils
import time
import torch.nn.functional as F
import shutil


class dep_model(nn.Module):
    def __init__(self, w2i, pos, options):
        super(dep_model, self).__init__()
        self.embedding_dim = options.wembedding_dim
        self.pdim = options.pembedding_dim
        self.hidden_dim = options.hidden_dim
        self.pre_output = options.pre_output
        self.feature_dim = options.feature_dim
        self.n_layer = options.n_layer
        self.external_embedding = options.external_embedding
        self.words = w2i
        self.pos = pos
        self.gpu = options.gpu
        self.dropout_ratio = options.dropout_ratio
        self.optim = options.optim
        self.order = options.order
        self.learning_rate = options.learning_rate

        self.lstm = nn.LSTM(self.embedding_dim + self.pdim, self.hidden_dim, self.n_layer, bidirectional=True,
                            batch_first=True)

        self.hidden2head = nn.Linear(2 * self.hidden_dim, self.feature_dim)
        self.hidden2modifier = nn.Linear(2 * self.hidden_dim, self.feature_dim)
        self.hidden2sibling = nn.Linear(2 * self.hidden_dim, self.feature_dim)
        self.hidden2grand = nn.Linear(2 * self.hidden_dim, self.feature_dim)

        self.sibling_preout = nn.Linear(3 * self.feature_dim, self.pre_output)
        self.grand_preout = nn.Linear(3 * self.feature_dim, self.pre_output)
        self.root_preout = nn.Linear(2 * self.feature_dim, self.pre_output)

        self.sibling_out = nn.Linear(self.pre_output, 1)
        self.grand_out = nn.Linear(self.pre_output, 1)

        self.plookup = nn.Embedding(len(pos), self.pdim)
        self.wlookup = nn.Embedding(len(self.words), self.embedding_dim)

        self.trainer = self.get_optim(self.parameters())

        self.tree_param = {}
        self.partition_table = {}
        self.encoder_score_table = {}
        self.parse_results = {}

    def get_optim(self, parameters):
        if self.optim == 'sgd':
            return optim.SGD(parameters, lr=self.learning_rate)
        elif self.optim == 'adam':
            return optim.Adam(parameters, lr=self.learning_rate)
        elif self.optim == 'adagrad':
            return optim.Adagrad(parameters, lr=self.learning_rate)
        elif self.optim == 'adadelta':
            return optim.Adadelta(parameters, lr=self.learning_rate)

    def evaluate_m1(self, hidden_out):
        batch_size, sentence_length = hidden_out.data.shape[0], hidden_out.data.shape[1]
        grand_scores = torch.zeros((batch_size, sentence_length, sentence_length, sentence_length), dtype=torch.float)
        sibling_scores = torch.zeros((batch_size, sentence_length, sentence_length, sentence_length), dtype=torch.float)
        if self.gpu > -1 and torch.cuda.is_available():
            grand_scores = grand_scores.cuda()
            sibling_scores = sibling_scores.cuda()
        grand_scores.fill_(-np.inf)
        # sibling_scores.fill_(-np.inf)
        for i in range(sentence_length):
            for j in range(sentence_length):
                for k in range(sentence_length):
                    if i == 0:
                        continue
                    if i == j or j == k:
                        # scoring for root
                        if k == 0 and j == k and i != j:
                            one_head_rep = self.hidden2head(hidden_out[:, j])
                            one_modifier_rep = self.hidden2modifier(hidden_out[:, i])
                            one_root_score = self.root_preout(F.tanh(torch.cat((one_head_rep, one_modifier_rep), 1)))
                            one_root_score = F.tanh(self.grand_out(one_root_score))
                            grand_scores[:, i, k, j] = one_root_score.view(-1)
                        else:
                            continue
                    else:
                        one_grand_rep = self.hidden2grand(hidden_out[:, k])
                        one_head_rep = self.hidden2head(hidden_out[:, j])
                        one_modifier_rep = self.hidden2modifier(hidden_out[:, i])

                        one_s_head_rep = self.hidden2head(hidden_out[:, k])
                        one_sibling_rep = self.hidden2sibling(hidden_out[:, j])
                        one_s_modifier_rep = self.hidden2modifier(hidden_out[:, i])

                        one_grand_score = self.grand_preout(
                            F.tanh(torch.cat((one_grand_rep, one_head_rep, one_modifier_rep), 1)))
                        one_sibling_score = self.sibling_preout(
                            F.tanh(torch.cat((one_s_head_rep, one_sibling_rep, one_s_modifier_rep), 1)))

                        one_grand_score = F.tanh(self.grand_out(one_grand_score))
                        one_sibling_score = F.tanh(self.sibling_out(one_sibling_score))

                        grand_scores[:, i, k, j] = one_grand_score.view(-1)
                        sibling_scores[:, i, k, j] = one_sibling_score.view(-1)

        return grand_scores, sibling_scores

    def check_gold(self, batch_parent):
        batch_size, sentence_length = batch_parent.shape
        g_heads = batch_parent
        g_heads_siblings = -torch.zeros((batch_size, sentence_length), dtype=torch.long)
        g_heads_grands = -torch.zeros((batch_size, sentence_length), dtype=torch.long)
        for s in range(batch_size):
            sibling_check = {}
            for i in range(1, sentence_length):
                head = g_heads[s, i]
                grand = g_heads[s, head] if head != 0 else 0
                g_heads_grands[s, i] = grand * sentence_length + head
                head_idx = head.item()
                if sibling_check.get(head_idx) is not None:
                    sibling_check[head_idx].append(i)
                else:
                    sibling_check[head_idx] = [i]
            for head, siblings in sibling_check.items():
                if len(siblings) == 1:
                    continue
                else:
                    for i, si in enumerate(siblings):
                        if i < len(siblings) - 1:
                            si_1 = siblings[i + 1]
                            if si < head and si_1 < head:
                                g_heads_siblings[s, si] = head * sentence_length + si_1
                            if si > head and si_1 > head:
                                g_heads_siblings[s, si_1] = head * sentence_length + si
        return g_heads_grands, g_heads_siblings, g_heads

    def construct_mask(self, batch_size, sentence_length):
        masks = np.zeros((batch_size, sentence_length, sentence_length, self.tag_num, self.tag_num))
        masks[:, :, 0, :, :] = 1
        if self.tag_num > 1:
            masks[:, 0, :, 1:, :] = 1
        # masks[:, :, :, :, 0] = 1
        for i in range(sentence_length):
            masks[:, i, i, :, :] = 1
        masks = masks.astype(int)
        mask_var = torch.ByteTensor(masks)
        return mask_var

    def forward(self, batch_words, batch_pos, batch_parent, batch_sen):
        batch_size, sentence_length = batch_words.data.size()
        w_embeds = self.wlookup(batch_words)
        p_embeds = self.plookup(batch_pos)
        # w_embeds = self.dropout1(w_embeds)
        batch_input = torch.cat((w_embeds, p_embeds), 2)
        hidden_out, _ = self.lstm(batch_input)
        if self.order == 3:
            grand_scores, sibling_scores = self.evaluate_m1(hidden_out)
            heads_grands, heads_siblings, heads = EL_M1.batch_parse(grand_scores, sibling_scores)
            g_heads_grands, g_heads_siblings, g_heads = self.check_gold(batch_parent)
            predicted_grands = torch.gather(
                grand_scores.view(batch_size, sentence_length, sentence_length * sentence_length), 2,
                heads_grands.view(batch_size, sentence_length, 1))
            predicted_siblings = torch.gather(
                sibling_scores.view(batch_size, sentence_length, sentence_length * sentence_length), 2,
                heads_siblings.view(batch_size, sentence_length, 1))
            golden_grands = torch.gather(
                grand_scores.view(batch_size, sentence_length, sentence_length * sentence_length), 2,
                g_heads_grands.view(batch_size, sentence_length, 1))
            golden_siblings = torch.gather(
                sibling_scores.view(batch_size, sentence_length, sentence_length * sentence_length), 2,
                heads_siblings.view(batch_size, sentence_length, 1))
            loss = predicted_grands[:, 1:] - golden_grands[:, 1:] + predicted_siblings[:, 1:] - golden_siblings[:, 1:]
            loss = torch.sum(loss)/batch_size
            return loss

    def save(self, fn):
        tmp = fn + '.tmp'
        torch.save(self.state_dict(), tmp)
        shutil.move(tmp, fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))
