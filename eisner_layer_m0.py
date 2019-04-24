# This file contains routines from Lisbon Machine Learning summer school.
# The code is freely distributed under a MIT license. https://github.com/LxMLS/lxmls-toolkit/
import numpy as np
import torch
from torch.nn.init import *
import utils


def batch_parse(batch_scores):
    batch_size, sentence_length, _, _ = batch_scores.shape
    # CYK table
    complete_table = torch.zeros((batch_size, sentence_length * sentence_length * sentence_length * 2), dtype=torch.float)
    incomplete_table = torch.zeros((batch_size, sentence_length * sentence_length * sentence_length * 2), dtype=torch.float)
    # backtrack table
    complete_backtrack = -torch.ones((batch_size, sentence_length * sentence_length * sentence_length * 2), dtype=torch.int)
    incomplete_backtrack = -torch.ones((batch_size, sentence_length * sentence_length * sentence_length * 2), dtype=torch.int)
    # span index table, to avoid redundant iterations
    span_2_id, id_2_span, ijkss, jimcs, kmjcs, jimis, kmjis, kimcs, imjcs, kimis, imjis, basic_span = utils.m0_constituent_index(sentence_length, False)
    # initial basic complete spans
    #if gpu>-1 and torch.cuda.is_available():
        #complete_table = complete_table.cuda()
        #incomplete_table = incomplete_table.cuda()

    for ijj in basic_span:
        complete_table[:, ijj] = 0.0

    for ijk in ijkss:
        (k, i, j, dir) = id_2_span[ijk]
        if dir == 0:
            num_jim = len(jimis[ijk])
            jim_ci = complete_table[:, jimis[ijk]].reshape(batch_size, num_jim)
            kmj_ci = complete_table[:, kmjis[ijk]].reshape(batch_size, num_jim)
            # construct incomplete spans
            span_i = jim_ci + kmj_ci + batch_scores[:, i, k, j].reshape(batch_size, 1)
            incomplete_table[:, ijk] = torch.max(span_i, dim=1)[0]
            max_idx = torch.max(span_i, dim=1)[1]
            incomplete_backtrack[:, ijk] = max_idx
            # construct complete spans
            num_jim = len(jimcs[ijk])
            jim_cc = complete_table[:, jimcs[ijk]].reshape(batch_size, num_jim)
            kmj_ic = incomplete_table[:, kmjcs[ijk]].reshape(batch_size, num_jim)
            span_c = jim_cc + kmj_ic
            complete_table[:, ijk] = torch.max(span_c, dim=1)[0]
            max_idx = torch.max(span_c, dim=1)[1]
            complete_backtrack[:, ijk] = max_idx
        else:
            num_kim = len(kimis[ijk])
            kim_ci = complete_table[:, kimis[ijk]].reshape(batch_size, num_kim)
            imj_ci = complete_table[:, imjis[ijk]].reshape(batch_size, num_kim)
            # construct incomplete spans
            span_i = kim_ci + imj_ci + batch_scores[:, j, k, i].reshape(batch_size, 1)
            incomplete_table[:, ijk] = torch.max(span_i, dim=1)[0]
            max_idx = torch.max(span_i, dim=1)[1]
            incomplete_backtrack[:, ijk] = max_idx

            num_kim = len(kimcs[ijk])
            kim_ic = incomplete_table[:, kimcs[ijk]].reshape(batch_size, num_kim)
            imj_cc = complete_table[:, imjcs[ijk]].reshape(batch_size, num_kim)
            # construct complete spans
            span_c = kim_ic + imj_cc
            complete_table[:, ijk] = torch.max(span_c, dim=1)[0]
            max_idx = torch.max(span_c, dim=1)[1]
            complete_backtrack[:, ijk] = max_idx

    heads = -torch.ones((batch_size, sentence_length), dtype=torch.long)
    heads_grand = -torch.zeros((batch_size, sentence_length), dtype=torch.long)
    root_id = span_2_id[(0, 0, sentence_length - 1, 1)]
    for s in range(batch_size):
        batch_backtracking(incomplete_backtrack, complete_backtrack, root_id, 1, heads, heads_grand,
                           jimcs, kmjcs, jimis, kmjis, kimcs, imjcs, kimis, imjis, id_2_span, span_2_id, s)

    return heads, heads_grand


def batch_backtracking(incomplete_backtrack, complete_backtrack, span_id, complete, heads, heads_grand,
                       jimcs, kmjcs, jimis, kmjis, kimcs, imjcs, kimis, imjis, id_2_span, span_2_id, sen_id):
    batch_size, sentence_length = heads.shape
    (k, i, j, dir) = id_2_span[span_id]
    if i == j:
        return
    if complete:
        if dir == 0:
            m = complete_backtrack[sen_id, span_id]
            left_span_id = jimcs[span_id][m]
            right_span_id = kmjcs[span_id][m]
            batch_backtracking(incomplete_backtrack, complete_backtrack, left_span_id, 1, heads, heads_grand,
                               jimcs, kmjcs, jimis, kmjis, kimcs, imjcs, kimis, imjis, id_2_span, span_2_id, sen_id)
            batch_backtracking(incomplete_backtrack, complete_backtrack, right_span_id, 0, heads, heads_grand,
                               jimcs, kmjcs, jimis, kmjis, kimcs, imjcs, kimis, imjis, id_2_span, span_2_id, sen_id)
            return
        else:
            m = complete_backtrack[sen_id, span_id]
            # print 'k is ', k, ' complete right'
            left_span_id = kimcs[span_id][m]
            right_span_id = imjcs[span_id][m]
            batch_backtracking(incomplete_backtrack, complete_backtrack, left_span_id, 0, heads, heads_grand,
                               jimcs, kmjcs, jimis, kmjis, kimcs, imjcs, kimis, imjis, id_2_span, span_2_id, sen_id)
            batch_backtracking(incomplete_backtrack, complete_backtrack, right_span_id, 1, heads, heads_grand,
                               jimcs, kmjcs, jimis, kmjis, kimcs, imjcs, kimis, imjis, id_2_span, span_2_id, sen_id)
            return
    else:
        if dir == 0:

            m = incomplete_backtrack[sen_id, span_id]
            # print 'k is ', k, ' incomplete left'
            heads[sen_id, i] = j
            heads_grand[sen_id, i] = k * sentence_length + j
            left_span_id = jimis[span_id][m]
            right_span_id = kmjis[span_id][m]
            batch_backtracking(incomplete_backtrack, complete_backtrack, left_span_id, 1, heads, heads_grand,
                               jimcs, kmjcs, jimis, kmjis, kimcs, imjcs, kimis, imjis, id_2_span, span_2_id, sen_id)
            batch_backtracking(incomplete_backtrack, complete_backtrack, right_span_id, 1, heads, heads_grand,
                               jimcs, kmjcs, jimis, kmjis, kimcs, imjcs, kimis, imjis, id_2_span, span_2_id, sen_id)
            return
        else:
            m = incomplete_backtrack[sen_id, span_id]
            # print 'k is', k, ' incomplete right'
            heads[sen_id, j] = i
            heads_grand[sen_id, j] = k * sentence_length + i
            left_span_id = kimis[span_id][m]
            right_span_id = imjis[span_id][m]
            batch_backtracking(incomplete_backtrack, complete_backtrack, left_span_id, 1, heads, heads_grand,
                               jimcs, kmjcs, jimis, kmjis, kimcs, imjcs, kimis, imjis, id_2_span, span_2_id, sen_id)
            batch_backtracking(incomplete_backtrack, complete_backtrack, right_span_id, 1, heads, heads_grand,
                               jimcs, kmjcs, jimis, kmjis, kimcs, imjcs, kimis, imjis, id_2_span, span_2_id, sen_id)
            return
