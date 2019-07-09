import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import *
from torch import optim
import eisner_layer_m1 as EL_M1
import eisner_layer_m0 as EL_M0
import eisner_layer as EL
import utils
import time
import torch.nn.functional as F
import shutil


class dep_model(nn.Module):
    def __init__(self, w2i, pos, options, feats=None):
        super(dep_model, self).__init__()
        self.embedding_dim = options.wembedding_dim
        self.pdim = options.pembedding_dim
        self.hidden_dim = options.hidden_dim
        self.feature_dim = options.feature_dim
        self.n_layer = options.n_layer
        self.external_embedding = options.external_embedding
        self.words = w2i
        self.pos = pos
        self.feats = feats
        self.gpu = options.gpu
        self.dropout_ratio = options.dropout_ratio
        self.optim = options.optim
        self.order = options.order
        self.learning_rate = options.learning_rate
        self.sparse_feature = options.sparse_feature
        self.combine_score = options.combine_score
        self.embedding_only = options.embedding_only

        if not self.sparse_feature:
            self.lstm = nn.LSTM(self.embedding_dim + self.pdim, self.hidden_dim, self.n_layer, bidirectional=True,
                                batch_first=True)

            self.hidden2head = nn.Linear(2 * self.hidden_dim, self.feature_dim)
            self.hidden2modifier = nn.Linear(2 * self.hidden_dim, self.feature_dim)
            self.hidden2sibling = nn.Linear(2 * self.hidden_dim, self.feature_dim)
            self.hidden2grand = nn.Linear(2 * self.hidden_dim, self.feature_dim)

            self.sibling_out = nn.Linear(self.feature_dim, 1)
            self.grand_out = nn.Linear(self.feature_dim, 1)
            self.head_out = nn.Linear(self.feature_dim, 1)
            self.modifier_out = nn.Linear(self.feature_dim, 1)

            self.feature_out = nn.Linear(self.feature_dim, 1)
            self.plookup = nn.Embedding(len(pos), self.pdim)
            self.wlookup = nn.Embedding(len(self.words), self.embedding_dim)
        else:
            self.feats_param = nn.Parameter(torch.FloatTensor(len(self.feats)))
            self.feats_param.data = torch.zeros(len(self.feats))

        if self.embedding_only:
            self.embed2hidden = nn.Linear(self.embedding_dim + self.pdim, 2*self.hidden_dim)

        self.trainer = self.get_optim(self.parameters())
        self.parse_results = {}

        self.ghms_scores_table = None
        self.hm_scores_table = None
        self.ghm_scores_table = None

    def get_optim(self, parameters):
        if self.optim == 'sgd':
            return optim.SGD(parameters, lr=self.learning_rate)
        elif self.optim == 'adam':
            return optim.Adam(parameters, lr=self.learning_rate)
        elif self.optim == 'adagrad':
            return optim.Adagrad(parameters, lr=self.learning_rate)
        elif self.optim == 'adadelta':
            return optim.Adadelta(parameters, lr=self.learning_rate)

    # scoring for first order model
    def evaluate_1st(self, hidden_out):
        head_score = self.hidden2head(hidden_out)
        batch_size, sentence_length, _ = hidden_out.shape
        head_score = self.feature_transform(head_score, 0, 1)

        modifier_score = self.hidden2modifier(hidden_out)
        modifier_score = self.feature_transform(modifier_score, 1, 1)

        self.hm_scores_table = head_score + modifier_score
        self.hm_scores_table = self.feature_out(torch.tanh(self.hm_scores_table))
        self.hm_scores_table = self.hm_scores_table.view(batch_size, sentence_length * sentence_length)
        return

    # scoring for first order feature based model
    def evaluate_1st_sparse(self, batch_feats):
        batch_size, sentence_length, _, feature_length = batch_feats.shape
        batch_scores = torch.index_select(self.feats_param, dim=0, index=batch_feats.contiguous().view(-1))
        batch_scores = batch_scores.view(batch_size, sentence_length, sentence_length, feature_length)
        batch_scores = torch.sum(batch_scores, dim=3)
        self.hm_scores_table = batch_scores.permute(0, 2, 1)
        self.hm_scores_table = self.hm_scores_table.contiguous().view(batch_size, sentence_length * sentence_length)
        return

    def evaluate_m0(self, hidden_out):
        batch_size, sentence_length, _ = hidden_out.shape
        head_score = self.hidden2head(hidden_out)
        head_score = self.feature_transform(head_score, 1, 2)
        modifier_score = self.hidden2modifier(hidden_out)
        modifier_score = self.feature_transform(modifier_score, 2, 2)
        grand_score = self.hidden2sibling(hidden_out)
        grand_score = self.feature_transform(grand_score, 0, 2)
        self.ghm_scores_table = head_score + modifier_score + grand_score
        self.ghm_scores_table = self.feature_out(torch.tanh(self.ghm_scores_table))
        if self.combine_score:
            self.evaluate_1st(hidden_out)
            self.ghm_scores_table = self.ghm_scores_table.view(batch_size, sentence_length, sentence_length, sentence_length) \
                                    + self.hm_scores_table.view(batch_size, sentence_length, sentence_length, 1)
        self.ghm_scores_table = self.ghm_scores_table.view(batch_size, sentence_length * sentence_length * sentence_length)
        return

    def evaluate_m0_sparse(self, batch_feats):
        batch_size, sentence_length, _, _, feature_length = batch_feats.shape
        batch_scores = torch.index_select(self.feats_param, dim=0, index=batch_feats.view(-1))
        batch_scores = batch_scores.view(batch_size, sentence_length, sentence_length, sentence_length, feature_length)
        batch_scores = torch.sum(batch_scores, dim=4)
        self.ghm_scores_table = batch_scores.permute(0, 3, 2, 1)
        if self.combine_score:
            batch_feats_1st = batch_feats[:, :, 0, :, 0:(batch_feats.shape[4] - 1)]
            self.evaluate_1st_sparse(batch_feats_1st)
            self.ghm_scores_table = self.ghm_scores_table.view(batch_size, sentence_length, sentence_length, sentence_length) \
                                    + self.hm_scores_table.view(batch_size, sentence_length, sentence_length, 1)
        self.ghm_scores_table = self.ghm_scores_table.contiguous().view(batch_size, sentence_length * sentence_length * sentence_length)
        return

    def evaluate_m1(self, hidden_out):
        # scores for grandparents
        grand_score = self.hidden2grand(hidden_out)
        grand_score = self.grand_out(grand_score)
        grand_score = self.feature_transform(grand_score, 2, 3)
        # scores for dependency head
        head_score = self.hidden2head(hidden_out)
        head_score = self.head_out(head_score)
        head_score = self.feature_transform(head_score, 1, 3)
        # scores for dependency child
        modifier_score = self.hidden2modifier(hidden_out)
        modifier_score = self.modifier_out(modifier_score)
        modifier_score = self.feature_transform(modifier_score, 3, 3)
        # scores for siblings
        sibling_score = self.hidden2sibling(hidden_out)
        sibling_score = self.sibling_out(sibling_score)
        sibling_score = self.feature_transform(sibling_score, 0, 3)

        # Concate all scores and add them together

        self.ghms_scores_table = torch.sum(torch.cat((modifier_score, grand_score, head_score, sibling_score), 2),
                                           dim=2)
        self.ghms_scores_table = torch.tanh(self.ghms_scores_table)

        return

    def evaluate_m1_sparse(self, batch_feats):
        batch_size, sentence_length, _, _, _, feature_length = batch_feats.shape
        batch_scores = torch.index_select(self.feats_param, dim=0, index=batch_feats.view(-1))
        batch_scores = batch_scores.view(batch_size, sentence_length, sentence_length, sentence_length, sentence_length, feature_length)
        batch_scores = torch.sum(batch_scores, dim=5)
        batch_scores = batch_scores.permute(0, 3, 2, 1, 4)
        self.ghms_scores_table = batch_scores
        return

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

    def check_gold_m0(self, batch_parent):
        batch_size, sentence_length = batch_parent.shape
        g_heads = batch_parent
        g_heads_grand = torch.zeros((batch_size, sentence_length), dtype=torch.long)
        for s in range(batch_size):
            for i in range(1, sentence_length):
                head = g_heads[s, i]
                grand = g_heads[s, head] if head != 0 else 0
                g_heads_grand[s, i] = grand * sentence_length + head
        return g_heads_grand, g_heads

    def forward(self, batch_words, batch_pos, batch_parent, batch_sen):
        batch_size, sentence_length = batch_words.data.size()
        if self.gpu > -1 and torch.cuda.is_available():
            batch_words = batch_words.cuda()
            batch_pos = batch_pos.cuda()
        w_embeds = self.wlookup(batch_words)
        p_embeds = self.plookup(batch_pos)
        # w_embeds = self.dropout1(w_embeds)
        batch_input = torch.cat((w_embeds, p_embeds), 2)
        if not self.embedding_only:
            hidden_out, _ = self.lstm(batch_input)
        else:
            hidden_out = self.embed2hidden(batch_input)

        if self.order == 1:
            self.evaluate_1st(hidden_out)
            if self.training:
                if self.gpu == -1:
                    self.hm_scores_table = self.hm_scores_table + torch.ones((batch_size, 1), dtype=torch.float)
                else:
                    self.hm_scores_table = self.hm_scores_table + torch.ones((batch_size, 1), dtype=torch.float).cuda()

                g_heads = batch_parent
                self.hm_scores_table = self.hm_scores_table.view(batch_size, sentence_length, sentence_length)
                for s in range(batch_size):
                    for i in range(1, sentence_length):
                        if self.gpu == -1:
                            self.hm_scores_table[s, i, g_heads[s, i]] = self.hm_scores_table[s, i, g_heads[s, i]] \
                                                                        - torch.ones((1), dtype=torch.float)
                        else:
                            self.hm_scores_table[s, i, g_heads[s, i]] = self.hm_scores_table[s, i, g_heads[s, i]] \
                                                                        - torch.ones((1), dtype=torch.float).cuda()
            else:
                g_heads = None
                self.hm_scores_table = self.hm_scores_table.view(batch_size, sentence_length, sentence_length)
            heads = EL.batch_parse(self.hm_scores_table.permute(0, 2, 1).cpu())

            if not self.training:
                for i in range(batch_size):
                    sen_idx = batch_sen[i]
                    self.parse_results[sen_idx] = heads[i]
                return
            heads[:, 0] = 0
            g_heads[:, 0] = 0
            if self.gpu > -1 and torch.cuda.is_available():
                heads = heads.cuda()
                g_heads = g_heads.cuda()
            predicted_hm = torch.gather(self.hm_scores_table, 2, heads.view((batch_size, sentence_length, 1)))
            golden_hm = torch.gather(self.hm_scores_table, 2, g_heads.view(batch_size, sentence_length, 1))
            loss = predicted_hm[:, 1:] - golden_hm[:, 1:]
            loss = torch.sum(loss) / batch_size
            return loss

        if self.order == 2:
            print '\n The sentence length in this batch is ' + str(sentence_length)
            start = time.clock()
            self.evaluate_m0(hidden_out)
            end = time.clock()
            print 'Time cost in evaluating scores: ' + str(end - start)
            if self.training:
                if self.gpu == -1:
                    self.ghm_scores_table = self.ghm_scores_table + torch.ones((batch_size, 1), dtype=torch.float)
                else:
                    self.ghm_scores_table = self.ghm_scores_table + torch.ones((batch_size, 1), dtype=torch.float).cuda()

                g_heads_grand, g_heads = self.check_gold_m0(batch_parent)
                self.ghm_scores_table = self.ghm_scores_table.view(batch_size, sentence_length, sentence_length, sentence_length)
                for s in range(batch_size):
                    for i in range(1, sentence_length):
                        if self.gpu == -1:
                            self.ghm_scores_table[s, i, :, g_heads[s, i]] = self.ghm_scores_table[s, i, :, g_heads[s, i]] \
                                                                            - torch.ones((1), dtype=torch.float)
                        else:
                            self.ghm_scores_table[s, i, :, g_heads[s, i]] = self.ghm_scores_table[s, i, :, g_heads[s, i]] \
                                                                            - torch.ones((1), dtype=torch.float).cuda()
            else:
                g_heads_grand = None
                self.ghm_scores_table = self.ghm_scores_table.view(batch_size, sentence_length, sentence_length, sentence_length)
            start = time.clock()
            heads, heads_grand = EL_M0.batch_parse(self.ghm_scores_table.cpu())
            end = time.clock()
            print 'Time cost in parsing: ' + str(end - start)
            if not self.training:
                for i in range(batch_size):
                    sen_idx = batch_sen[i]
                    self.parse_results[sen_idx] = heads[i]
                return

            if self.gpu > -1 and torch.cuda.is_available():
                heads_grand = heads_grand.cuda()
                g_heads_grand = g_heads_grand.cuda()
            predicted_ghm = torch.gather(
                self.ghm_scores_table.view(batch_size, sentence_length, sentence_length * sentence_length), 2,
                heads_grand.view(batch_size, sentence_length, 1))
            golden_ghm = torch.gather(self.ghm_scores_table.view(batch_size, sentence_length, sentence_length * sentence_length), 2,
                                      g_heads_grand.view(batch_size, sentence_length, 1))
            loss = predicted_ghm[:, 1:] - golden_ghm[:, 1:]
            loss = torch.sum(loss) / batch_size
            return loss
        if self.order == 3:
            # start = time.clock()
            self.evaluate_m1(hidden_out)
            if self.training:

                # Add the margin for all the scores

                if self.gpu == -1:
                    self.ghms_scores_table = self.ghms_scores_table + torch.ones((batch_size, 1), dtype=torch.float)
                else:
                    self.ghms_scores_table = self.ghms_scores_table + torch.ones((batch_size, 1), dtype=torch.float).cuda()
                # elasped = time.clock() - start
                # print "time cost in computing score " + str(elasped)

                g_heads_grand_sibling, g_heads = self.check_gold(batch_parent)
                self.ghms_scores_table = self.ghms_scores_table.view(batch_size, sentence_length, sentence_length, sentence_length, sentence_length)

                # Remove the margin for golden parse
                for s in range(batch_size):
                    for i in range(sentence_length):
                        if self.gpu == -1:
                            self.ghms_scores_table[s, i, :, g_heads[s, i], :] = self.ghms_scores_table[s, i, :, g_heads[s, i], :] \
                                                                                - torch.ones((1), dtype=torch.float)
                        else:
                            self.ghms_scores_table[s, i, :, g_heads[s, i], :] = self.ghms_scores_table[s, i, :, g_heads[s, i], :] \
                                                                                - torch.ones((1), dtype=torch.float).cuda()
            else:
                g_heads_grand_sibling = None
                self.ghms_scores_table = self.ghms_scores_table.view(batch_size, sentence_length, sentence_length,
                                                                     sentence_length, sentence_length)
            # start = time.clock()
            heads_grand_sibling, heads, _ = EL_M1.batch_parse(self.ghms_scores_table.cpu())

            if not self.training:
                for i in range(batch_size):
                    sen_idx = batch_sen[i]
                    self.parse_results[sen_idx] = heads[i]
                return

            if self.gpu > -1 and torch.cuda.is_available():
                heads_grand_sibling = heads_grand_sibling.cuda()
                g_heads_grand_sibling = g_heads_grand_sibling.cuda()
            # elasped = time.clock() - start
            # print "time cost in parsing " + str(elasped)
            predicted_ghms = torch.gather(
                self.ghms_scores_table.view(batch_size, sentence_length,
                                            sentence_length * sentence_length * sentence_length), 2,
                heads_grand_sibling.view(batch_size, sentence_length, 1))
            golden_ghms = torch.gather(self.ghms_scores_table.view(batch_size, sentence_length,
                                                                   sentence_length * sentence_length * sentence_length), 2,
                                       g_heads_grand_sibling.view(batch_size, sentence_length, 1))
            loss_compare = torch.zeros((batch_size, 1), dtype=torch.float)
            if self.gpu > -1 and torch.cuda.is_available():
                loss_compare = loss_compare.cuda()
            loss = predicted_ghms[:, 1:] - golden_ghms[:, 1:]
            # loss = torch.max(torch.sum(loss, dim=1), loss_compare)
            loss = torch.sum(loss) / batch_size
            return loss

    # Sparse feature version of forward function
    def forward_sparse(self, batch_parent, batch_feats, batch_sen):
        if self.gpu > -1 and torch.cuda.is_available():
            batch_feats = batch_feats.cuda()
        if self.order == 1:
            batch_size, sentence_length, _, _ = batch_feats.size()
            self.evaluate_1st_sparse(batch_feats)
            if self.training:
                if self.gpu == -1:
                    self.hm_scores_table = self.hm_scores_table + torch.ones((batch_size, 1), dtype=torch.float)
                else:
                    self.hm_scores_table = self.hm_scores_table + torch.ones((batch_size, 1), dtype=torch.float).cuda()
                g_heads = batch_parent
                self.hm_scores_table = self.hm_scores_table.view(batch_size, sentence_length, sentence_length)
                # Remove margin for golden parse
                for s in range(batch_size):
                    for i in range(1, sentence_length):
                        if self.gpu == -1:
                            self.hm_scores_table[s, i, g_heads[s, i]] = self.hm_scores_table[s, i, g_heads[s, i]] \
                                                                        - torch.ones((1), dtype=torch.float)
                        else:
                            self.hm_scores_table[s, i, g_heads[s, i]] = self.hm_scores_table[s, i, g_heads[s, i]] \
                                                                        - torch.ones((1), dtype=torch.float).cuda()
            else:
                g_heads = None
                self.hm_scores_table = self.hm_scores_table.view(batch_size, sentence_length, sentence_length)
            heads = EL.batch_parse(self.hm_scores_table.permute(0, 2, 1).cpu())
            if not self.training:
                for i in range(batch_size):
                    sen_idx = batch_sen[i]
                    self.parse_results[sen_idx] = heads[i]
                return
            if self.gpu > -1 and torch.cuda.is_available():
                heads = heads.cuda()
                g_heads = g_heads.cuda()
            heads[:, 0] = 0
            g_heads[:, 0] = 0
            predicted_hm = torch.gather(self.hm_scores_table, 2, heads.view(batch_size, sentence_length, 1))
            golden_hm = torch.gather(self.hm_scores_table, 2, g_heads.view(batch_size, sentence_length, 1))
            loss = predicted_hm[:, 1:] - golden_hm[:, 1:]
            loss = torch.sum(loss) / batch_size
            return loss

        if self.order == 2:
            batch_size, sentence_length, _, _, _ = batch_feats.size()
            print '\n The sentence length in this batch is ' + str(sentence_length)
            start = time.clock()
            self.evaluate_m0_sparse(batch_feats)
            end = time.clock()
            print 'Time cost in evaluating scores ' + str(end - start)
            if self.training:
                if self.gpu == -1:
                    self.ghm_scores_table = self.ghm_scores_table + torch.ones((batch_size, 1), dtype=torch.float)
                else:
                    self.ghm_scores_table = self.ghm_scores_table + torch.ones((batch_size, 1), dtype=torch.float).cuda()
                g_heads_grand, g_heads = self.check_gold_m0(batch_parent)
                self.ghm_scores_table = self.ghm_scores_table.view(batch_size, sentence_length, sentence_length, sentence_length)
                # Remove margin for golden parse
                for s in range(batch_size):
                    for i in range(sentence_length):
                        if self.gpu == -1:
                            self.ghm_scores_table[s, i, :, g_heads[s, i]] = self.ghm_scores_table[s, i, :, g_heads[s, i]] \
                                                                            - torch.ones((1), dtype=torch.float)
                        else:
                            self.ghm_scores_table[s, i, :, g_heads[s, i]] = self.ghm_scores_table[s, i, :, g_heads[s, i]] \
                                                                            - torch.ones((1), dtype=torch.float).cuda()
            else:
                g_heads_grand = None
                self.ghm_scores_table = self.ghm_scores_table.view(batch_size, sentence_length, sentence_length, sentence_length)
            start = time.clock()
            heads, heads_grand = EL_M0.batch_parse(self.ghm_scores_table.cpu())
            end = time.clock()
            print 'Time cost in parsing: ' + str(end - start)
            if not self.training:
                for i in range(batch_size):
                    sen_idx = batch_sen[i]
                    self.parse_results[sen_idx] = heads[i]
                return
            if self.gpu > -1 and torch.cuda.is_available():
                heads_grand = heads_grand.cuda()
                g_heads_grand = g_heads_grand.cuda()

            predicted_ghm = torch.gather(self.ghm_scores_table.contiguous().view(batch_size, sentence_length, sentence_length * sentence_length), 2,
                                         heads_grand.view(batch_size, sentence_length, 1))
            golden_ghm = torch.gather(self.ghm_scores_table.contiguous().view(batch_size, sentence_length, sentence_length * sentence_length), 2,
                                      g_heads_grand.view(batch_size, sentence_length, 1))
            loss = predicted_ghm[:, 1:] - golden_ghm[:, 1:]
            loss = torch.sum(loss) / batch_size
            return loss
        if self.order == 3:
            batch_size, sentence_length, _, _, _, _ = batch_feats.size()
            self.evaluate_m1_sparse(batch_feats)
            if self.training:
                if self.gpu == -1:
                    self.ghms_scores_table = self.ghms_scores_table + torch.ones((batch_size, 1), dtype=torch.float)
                else:
                    self.ghms_scores_table = self.ghms_scores_table + torch.ones((batch_size, 1), dtype=torch.float).cuda()
                g_heads_grand_sibling, g_heads = self.check_gold(batch_parent)

                # Remove margin for golden parse
                for s in range(batch_size):
                    for i in range(sentence_length):
                        if self.gpu == -1:
                            self.ghms_scores_table[s, i, :, g_heads[s, i], :] = self.ghms_scores_table[s, i, :, g_heads[s, i], :] \
                                                                                - torch.ones((1), dtype=torch.float)
                        else:
                            self.ghms_scores_table[s, i, :, g_heads[s, i], :] = self.ghms_scores_table[s, i, :, g_heads[s, i], :] \
                                                                                - torch.ones((1), dtype=torch.float).cuda()
            else:
                g_heads_grand_sibling = None
            heads_grand_sibling, heads, _ = EL_M1.batch_parse(self.ghms_scores_table.cpu())
            if not self.training:
                for i in range(batch_size):
                    sen_idx = batch_sen[i]
                    self.parse_results[sen_idx] = heads[i]
                return

            predicted_ghms = torch.gather(
                self.ghms_scores_table.contiguous().view(batch_size, sentence_length,
                                                         sentence_length * sentence_length * sentence_length), 2,
                heads_grand_sibling.view(batch_size, sentence_length, 1))
            golden_ghms = torch.gather(self.ghms_scores_table.contiguous().view(batch_size, sentence_length,
                                                                                sentence_length * sentence_length * sentence_length), 2,
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
