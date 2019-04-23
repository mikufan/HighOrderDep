# This file contains routines from Lisbon Machine Learning summer school.
# The code is freely distributed under a MIT license. https://github.com/LxMLS/lxmls-toolkit/
import numpy as np
import torch
from torch.nn.init import *
import utils


def batch_parse(batch_scores):
    batch_size, sentence_length, _ = batch_scores.shape
    # CYK table
    complete_table = torch.zeros((batch_size, sentence_length * sentence_length * 2), dtype=torch.float)
    incomplete_table = torch.zeros((batch_size, sentence_length * sentence_length * 2), dtype=torch.float)
    # backtrack table
    complete_backtrack = -torch.ones((batch_size, sentence_length * sentence_length * 2), dtype=torch.int)
    incomplete_backtrack = -torch.ones((batch_size, sentence_length * sentence_length * 2), dtype=torch.int)
    # span index table, to avoid redundant iterations
    span_2_id, id_2_span, ijss, ikcs, ikis, kjcs, kjis, basic_span = utils.constituent_index(sentence_length, False)
    # initial basic complete spans
    for ii in basic_span:
        complete_table[:, ii] = 0.0

    for ij in ijss:
        (l, r, dir) = id_2_span[ij]
        num_ki = len(ikis[ij])
        ik_ci = complete_table[:, ikis[ij]].reshape(batch_size, num_ki)
        kj_ci = complete_table[:, kjis[ij]].reshape(batch_size, num_ki)
        # construct incomplete spans
        if dir == 0:
            span_i = ik_ci + kj_ci + batch_scores[:, r, l].reshape(batch_size, 1)
        else:
            span_i = ik_ci + kj_ci + batch_scores[:, l, r].reshape(batch_size, 1)

        incomplete_table[:, ij] = torch.max(span_i, dim=1)[0]
        max_idx = torch.max(span_i, dim=1)[1]
        incomplete_backtrack[:, ij] = max_idx

        num_kc = len(ikcs[ij])
        if dir == 0:
            ik_cc = complete_table[:, ikcs[ij]].reshape(batch_size, num_kc)
            kj_ic = incomplete_table[:, kjcs[ij]].reshape(batch_size, num_kc)
            span_c = ik_cc + kj_ic
        else:
            ik_ic = incomplete_table[:, ikcs[ij]].reshape(batch_size, num_kc)
            kj_cc = complete_table[:, kjcs[ij]].reshape(batch_size, num_kc)
            span_c = ik_ic + kj_cc
        complete_table[:, ij] = torch.max(span_c, dim=1)[0]
        max_idx = torch.max(span_c, dim=1)[1]
        complete_backtrack[:, ij] = max_idx

    heads = -torch.ones((batch_size, sentence_length),dtype=torch.long)
    root_id = span_2_id[(0, sentence_length - 1, 1)]
    for s in range(batch_size):
        batch_backtracking(incomplete_backtrack, complete_backtrack, root_id, 1, heads,
                           ikcs, ikis, kjcs, kjis, id_2_span, span_2_id, s)

    return heads


def batch_backtracking(incomplete_backtrack, complete_backtrack, span_id, complete, heads,
                       ikcs, ikis, kjcs, kjis, id_2_span, span_2_id, sen_id):
    # print span_id
    (l, r, dir) = id_2_span[span_id]
    if l == r:
        return
    if complete:
        if dir == 0:
            k = complete_backtrack[sen_id, span_id]
            # print 'k is ', k, ' complete left'
            left_span_id = ikcs[span_id][k]
            right_span_id = kjcs[span_id][k]
            batch_backtracking(incomplete_backtrack, complete_backtrack, left_span_id, 1, heads,
                               ikcs, ikis, kjcs, kjis, id_2_span, span_2_id, sen_id)
            batch_backtracking(incomplete_backtrack, complete_backtrack, right_span_id, 0, heads,
                               ikcs, ikis, kjcs, kjis, id_2_span, span_2_id, sen_id)
            return
        else:
            k = complete_backtrack[sen_id, span_id]
            # print 'k is ', k, ' complete right'
            left_span_id = ikcs[span_id][k]
            right_span_id = kjcs[span_id][k]
            batch_backtracking(incomplete_backtrack, complete_backtrack, left_span_id, 0, heads,
                               ikcs, ikis, kjcs, kjis, id_2_span, span_2_id, sen_id)
            batch_backtracking(incomplete_backtrack, complete_backtrack, right_span_id, 1, heads,
                               ikcs, ikis, kjcs, kjis, id_2_span, span_2_id, sen_id)
            return
    else:
        if dir == 0:

            k = incomplete_backtrack[sen_id, span_id]
            # print 'k is ', k, ' incomplete left'
            heads[sen_id, l] = r
            left_span_id = ikis[span_id][k]
            right_span_id = kjis[span_id][k]
            batch_backtracking(incomplete_backtrack, complete_backtrack, left_span_id, 1, heads,
                               ikcs, ikis, kjcs, kjis, id_2_span, span_2_id, sen_id)
            batch_backtracking(incomplete_backtrack, complete_backtrack, right_span_id, 1, heads,
                               ikcs, ikis, kjcs, kjis, id_2_span, span_2_id, sen_id)
            return
        else:
            k = incomplete_backtrack[sen_id, span_id]
            # print 'k is', k, ' incomplete right'
            heads[sen_id, r] = l
            left_span_id = ikis[span_id][k]
            right_span_id = kjis[span_id][k]
            batch_backtracking(incomplete_backtrack, complete_backtrack, left_span_id, 1, heads,
                               ikcs, ikis, kjcs, kjis, id_2_span, span_2_id, sen_id)
            batch_backtracking(incomplete_backtrack, complete_backtrack, right_span_id, 1, heads,
                               ikcs, ikis, kjcs, kjis, id_2_span, span_2_id, sen_id)
            return
