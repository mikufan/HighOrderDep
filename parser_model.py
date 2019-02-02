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

        # self.sibling_preout = nn.Linear(3 * self.feature_dim, self.pre_output)
        # self.grand_preout = nn.Linear(3 * self.feature_dim, self.pre_output)
        # self.root_preout = nn.Linear(2 * self.feature_dim, self.pre_output)
        self.ghms_preout = nn.Linear(4 * self.feature_dim, self.pre_output)

        self.sibling_out = nn.Linear(self.pre_output, 1)
        self.grand_out = nn.Linear(self.pre_output, 1)
        self.ghms_out = nn.Linear(self.pre_output, 1)

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
        ghms_scores = torch.zeros((batch_size, sentence_length, sentence_length, sentence_length, sentence_length),
                                  dtype=torch.float)
        if self.gpu > -1 and torch.cuda.is_available():
            ghms_scores = ghms_scores.cuda()
        # ghms_scores.fill_(-np.inf)
        grand_score = self.hidden2grand(hidden_out)
        grand_score = self.feature_transform(grand_score, 0, 3)
        head_score = self.hidden2head(hidden_out)
        head_score = self.feature_transform(head_score, 1, 3)
        modifier_score = self.hidden2modifier(hidden_out)
        modifier_score = self.feature_transform(modifier_score, 2, 3)
        sibling_score = self.hidden2sibling(hidden_out)
        sibling_score = self.feature_transform(sibling_score, 3, 3)
        ghms_scores_table = self.ghms_preout(
            F.tanh(torch.cat((grand_score, head_score, modifier_score, sibling_score), 2)))
        ghms_scores_table = F.tanh(self.ghms_out(ghms_scores_table))
        ghms_scores_table = ghms_scores_table.view(batch_size, pow(sentence_length, 4))
        start = time.clock()
        for i in range(sentence_length):
            for j in range(sentence_length):
                if j == 0:
                    continue
                if i == j:
                    continue
                left = i if j > i else j
                right = j if j > i else i
                for k in range(sentence_length):
                    if left < k < right:
                        continue
                    for m in range(left, right + 1):
                        # one_grand_rep = grand_score[:, k]
                        # one_head_rep = head_score[:, i]
                        # one_modifier_rep = modifier_score[:, j]
                        # one_sibling_rep = sibling_score[:, m]
                        # one_ghms_score = self.ghms_preout(
                        # F.tanh(torch.cat((one_grand_rep, one_head_rep, one_modifier_rep, one_sibling_rep), 1)))
                        # one_ghms_score = F.tanh(self.ghms_out(one_ghms_score))
                        score_index = m * pow(sentence_length, 3) + i * pow(sentence_length,
                                                                            2) + j * sentence_length + k
                        ghms_scores[:, j, k, i, m] = ghms_scores_table[:, score_index]
        elasped = time.clock() - start
        print "time cost in iteration " + str(elasped)
        return ghms_scores

    def check_gold(self, batch_parent):
        batch_size, sentence_length = batch_parent.shape
        g_heads = batch_parent
        g_heads_grand_sibling = torch.zeros((batch_size, sentence_length), dtype=torch.long)
        grands = torch.zeros((batch_size, sentence_length), dtype=torch.long)
        for s in range(batch_size):
            sibling_check = {}
            for i in range(1, sentence_length):
                head = g_heads[s, i]
                grand = g_heads[s, head] if head != 0 else 0
                grands[s, i] = grand
                head_idx = head.item()
                if sibling_check.get(head_idx) is not None:
                    sibling_check[head_idx].append(i)
                else:
                    sibling_check[head_idx] = [i]
            for head, siblings in sibling_check.items():
                if len(siblings) == 1:
                    si = siblings[0]
                    grand = grands[s, si]
                    g_heads_grand_sibling[
                        s, si] = grand * sentence_length * sentence_length + head * sentence_length + head
                else:
                    for i, si in enumerate(siblings):
                        if i < len(siblings) - 1:
                            si_1 = siblings[i + 1]
                            if si < head and si_1 < head:
                                grand = grands[s, si]
                                g_heads_grand_sibling[
                                    s, si] = grand * sentence_length * sentence_length + head * sentence_length + si_1
                            if si > head and si_1 > head:
                                grand = grands[s, si_1]
                                g_heads_grand_sibling[
                                    s, si_1] = grand * sentence_length * sentence_length + head * sentence_length + si
                            if si < head and si_1 > head:
                                grand = grands[s, si]
                                g_heads_grand_sibling[
                                    s, si] = grand * sentence_length * sentence_length + head * sentence_length + head
                                grand = grands[s, si_1]
                                g_heads_grand_sibling[
                                    s, si_1] = grand * sentence_length * sentence_length + head * sentence_length + head
                            if i == 0 and si > head:
                                grand = grands[s, si]
                                g_heads_grand_sibling[
                                    s, si] = grand * sentence_length * sentence_length + head * sentence_length + head
                            if (i + 1) == len(siblings) - 1 and si_1 < head:
                                grand = grands[s, si_1]
                                g_heads_grand_sibling[
                                    s, si_1] = grand * sentence_length * sentence_length + head * sentence_length + head

        return g_heads_grand_sibling, g_heads

    def forward(self, batch_words, batch_pos, batch_parent, batch_sen):
        batch_size, sentence_length = batch_words.data.size()
        if self.gpu > -1 and torch.cuda.is_available():
            batch_words = batch_words.cuda()
            batch_pos = batch_pos.cuda()
        w_embeds = self.wlookup(batch_words)
        p_embeds = self.plookup(batch_pos)
        # w_embeds = self.dropout1(w_embeds)
        batch_input = torch.cat((w_embeds, p_embeds), 2)
        hidden_out, _ = self.lstm(batch_input)
        if self.order == 3:
            start = time.clock()
            ghms_scores = self.evaluate_m1(hidden_out)
            elasped = time.clock() - start
            print "time cost in computing score " + str(elasped)
            start = time.clock()
            g_heads_grand_sibling, g_heads = self.check_gold(batch_parent)
            elasped = time.clock() - start
            print "time cost in finding gold " + str(elasped)
            start = time.clock()
            heads_grand_sibling, heads = EL_M1.batch_parse(ghms_scores, g_heads_grand_sibling, self.gpu)
            #heads_grand_sibling = torch.ones((batch_size, pow(sentence_length,4)), dtype=torch.long)
            #g_heads_grand_sibling = 2 * torch.ones((batch_size, pow(sentence_length,4)), dtype=torch.long)
            elasped = time.clock() - start
            print "time cost in parsing " + str(elasped)
            # print(heads)
            batch_base = torch.zeros((batch_size, sentence_length), dtype=torch.long)
            if self.gpu > -1 and torch.cuda.is_available():
                 heads_grand_sibling = heads_grand_sibling.cuda()
                 g_heads_grand_sibling = g_heads_grand_sibling.cuda()
                 batch_base = batch_base.cuda()
            batch_margin = torch.ne((g_heads_grand_sibling - heads_grand_sibling), batch_base).type(torch.float)
            if self.gpu > -1 and torch.cuda.is_available():
                batch_margin = batch_margin.cuda()
            predicted_ghms = torch.gather(
                ghms_scores.view(batch_size, sentence_length, sentence_length * sentence_length * sentence_length), 2,
                heads_grand_sibling.view(batch_size, sentence_length, 1))
            predicted_ghms = predicted_ghms + batch_margin.view(batch_size, sentence_length, 1)
            golden_ghms = torch.gather(
                ghms_scores.view(batch_size, sentence_length, sentence_length * sentence_length * sentence_length), 2,
                g_heads_grand_sibling.view(batch_size, sentence_length, 1))
            loss = predicted_ghms[:, 1:] - golden_ghms[:, 1:]
            loss = torch.sum(loss) / batch_size
            return loss

    def feature_transform(self, score, index, highest_index):
        batch_size, sentence_length, dim = score.shape
        expand_time = pow(sentence_length, index)
        score = score.view(batch_size, sentence_length, 1, dim)
        score = score.expand(-1, -1, expand_time, -1)
        score = score.contiguous().view(batch_size, sentence_length * expand_time, dim)
        repeat_time = pow(sentence_length, highest_index) / expand_time
        score = score.repeat(1, repeat_time, 1)
        return score

    def save(self, fn):
        tmp = fn + '.tmp'
        torch.save(self.state_dict(), tmp)
        shutil.move(tmp, fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))
