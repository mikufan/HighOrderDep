import torch
import utils
import numpy as np


def batch_parse(batch_sibling_scores, batch_grand_scores):
    batch_size, sentence_length, _, _ = batch_sibling_scores.shape
    # CYK table
    complete_table = torch.zeros((batch_size, sentence_length * sentence_length * sentence_length * 2),
                                 dtype=torch.float)
    incomplete_table = torch.zeros((batch_size, sentence_length * sentence_length * sentence_length * 2),
                                   dtype=torch.float)
    sibling_table = torch.zeros((batch_size, sentence_length * sentence_length * sentence_length * 2),
                                dtype=torch.float)
    complete_table.fill_(-np.inf)
    incomplete_table.fill_(-np.inf)
    sibling_table.fill_(-np.inf)
    # backtrack table
    complete_backtrack = -torch.ones((batch_size, sentence_length * sentence_length * sentence_length * 2),
                                     dtype=torch.int)
    incomplete_backtrack = -torch.ones((batch_size, sentence_length * sentence_length * sentence_length * 2),
                                       dtype=torch.int)
    sibling_backtrack = -torch.ones((batch_size, sentence_length * sentence_length * sentence_length * 2),
                                    dtype=torch.int)
    # span index table, to avoid redundant iterations
    span_2_id, id_2_span, ijkss, jimcs, kmjcs, jimis, kmjis, jimics, kmjics, kimsib, kmjsib, \
    kimcs, imjcs, kimis, imjis, kimics, imjics, basic_span, id_2_sib = utils.m1_constituent_index(sentence_length,
                                                                                                  False)
    # initial basic complete spans
    for ijj in basic_span:
        complete_table[:, ijj] = 0.0

    for ijk in ijkss:
        (k, i, j, dir) = id_2_span[ijk]
        if dir == 0:
            # construct complete spans
            num_jim = len(jimcs[ijk])
            if num_jim > 0:
                jim_cc = complete_table[:, jimcs[ijk]].view(batch_size, num_jim)
                kmj_ic = incomplete_table[:, kmjcs[ijk]].view(batch_size, num_jim)
                span_c = jim_cc + kmj_ic
                complete_table[:, ijk] = torch.max(span_c, dim=1)[0]
                complete_backtrack[:, ijk] = torch.max(span_c, dim=1)[1]
            # construct incomplete spans
            num_jimc = len(jimics[ijk])
            if num_jimc > 0:
                jim_ci = complete_table[:, jimics[ijk]].view(batch_size, 1)
                kmj_ci = complete_table[:, kmjics[ijk]].view(batch_size, 1)
                span_ci = jim_ci + kmj_ci + batch_grand_scores[:, k, j, j, i].view(batch_size, 1)
                num_jim = len(jimis[ijk])
                span_i = span_ci
                if num_jim > 0:
                    jim_sibi = sibling_table[:, jimis[ijk]].view(batch_size, num_jim)
                    kmj_ii = incomplete_table[:, kmjis[ijk]].view(batch_size, num_jim)
                    sibs = id_2_sib
                    span_ii = jim_sibi + kmj_ii + batch_grand_scores[:, k, j, sibs, i].view(batch_size, num_jim)
                    span_i = torch.cat((span_ii, span_i), dim=1)
                incomplete_table[:, ijk] = torch.max(span_i, dim=1)[0]
                incomplete_backtrack[:, ijk] = torch.max(span_i, dim=1)[1]
            # construct sibling spans
            num_kim = len(kimsib[ijk])
            if num_kim > 0:
                kim_csib = complete_table[:, kimsib[ijk]].view(batch_size, num_kim)
                kmj_csib = complete_table[:, kmjsib[ijk]].view(batch_size, num_kim)
                span_s = kim_csib + kmj_csib
                sibling_table[:, ijk] = torch.max(span_s, dim=1)[0]
                sibling_backtrack[:, ijk] = torch.max(span_s, dim=1)[1]
        else:
            # construct complete spans
            num_kim = len(kimcs[ijk])
            if num_kim > 0:
                kim_ic = incomplete_table[:, kimcs[ijk]].view(batch_size, num_kim)
                imj_cc = complete_table[:, imjcs[ijk]].view(batch_size, num_kim)
                span_c = kim_ic + imj_cc
                complete_table[:, ijk] = torch.max(span_c, dim=1)[0]
                complete_backtrack[:, ijk] = torch.max(span_c, dim=1)[1]
            # construct incomplete spans
            num_kimc = len(kimics[ijk])
            if num_kimc > 0:
                kim_ci = complete_table[:, kimics[ijk]].view(batch_size, 1)
                imj_ci = complete_table[:, kmjics[ijk]].view(batch_size, 1)
                span_ci = jim_ci + kmj_ci + batch_grand_scores[:, k, j, j, i].view(batch_size, 1)
                num_jim = len(jimis[ijk])
                kim_ii = incomplete_table[:, kimis[ijk]].view(batch_size, num_kim)
                imj_sibi = sibling_table[:, imjis[ijk]].view(batch_size, num_kim)
                span_i = kim_ii + imj_sibi + batch_grand_scores[:, k, i, j].view(batch_size, 1)
                incomplete_table[:, ijk] = torch.max(span_i, dim=1)[0]
                incomplete_backtrack[:, ijk] = torch.max(span_i, dim=1)[1]

            # construct sibling spans
            num_kim = len(kimsib[ijk])
            if num_kim > 0:
                kim_csib = complete_table[:, kimsib[ijk]].view(batch_size, num_kim)
                kmj_csib = complete_table[:, kmjsib[ijk]].view(batch_size, num_kim)
                span_s = kim_csib + kmj_csib + batch_sibling_scores[:, k, i, j].view(batch_size, 1)
                sibling_table[:, ijk] = torch.max(span_s, dim=1)[0]
                sibling_backtrack[:, ijk] = torch.max(span_s, dim=1)[1]
    final_span = (0, 0, sentence_length - 1, 1)
    root_id = span_2_id[final_span]

    heads_grands = torch.zeros((batch_size, sentence_length), dtype=torch.long)
    heads_siblings = torch.zeros((batch_size, sentence_length), dtype=torch.long)
    heads = -torch.ones((batch_size, sentence_length), dtype=torch.long)
    for s in range(batch_size):
        batch_backtracking(incomplete_backtrack, complete_backtrack, sibling_backtrack, root_id, 0, heads_grands,
                           heads_siblings, heads, ijkss, jimcs, kmjcs, jimis, kmjis, kimsib, kmjsib, kimcs, imjcs,
                           kimis, imjis, id_2_span, s)

    return heads_grands, heads_siblings, heads


def batch_backtracking(incomplete_backtrack, complete_backtrack, sibling_backtrack, span_id, span_type, heads_grands,
                       heads_siblings, heads, ijkss, jimcs, kmjcs, jimis, kmjis, kimsib, kmjsib, kimcs, imjcs,
                       kimis, imjis, id_2_span, sen_id):
    batch_size, sentence_length = heads.shape
    # span_type 0 for complete, 1 for incomplete, 2 for sibling
    (k, i, j, dir) = id_2_span[span_id]
    if span_type == 0:
        if i == j:
            return
        else:
            m = complete_backtrack[sen_id, span_id]
            if dir == 0:
                left_span_id = jimcs[span_id][m]
                right_span_id = kmjcs[span_id][m]
                batch_backtracking(incomplete_backtrack, complete_backtrack, sibling_backtrack, left_span_id, 0,
                                   heads_grands,
                                   heads_siblings, heads, ijkss, jimcs, kmjcs, jimis, kmjis, kimsib, kmjsib, kimcs,
                                   imjcs,
                                   kimis, imjis, id_2_span, sen_id)
                batch_backtracking(incomplete_backtrack, complete_backtrack, sibling_backtrack, right_span_id, 1,
                                   heads_grands,
                                   heads_siblings, heads, ijkss, jimcs, kmjcs, jimis, kmjis, kimsib, kmjsib, kimcs,
                                   imjcs,
                                   kimis, imjis, id_2_span, sen_id)
            else:
                left_span_id = kimcs[span_id][m]
                right_span_id = imjcs[span_id][m]
                batch_backtracking(incomplete_backtrack, complete_backtrack, sibling_backtrack, left_span_id, 1,
                                   heads_grands,
                                   heads_siblings, heads, ijkss, jimcs, kmjcs, jimis, kmjis, kimsib, kmjsib, kimcs,
                                   imjcs,
                                   kimis, imjis, id_2_span, sen_id)
                batch_backtracking(incomplete_backtrack, complete_backtrack, sibling_backtrack, right_span_id, 0,
                                   heads_grands,
                                   heads_siblings, heads, ijkss, jimcs, kmjcs, jimis, kmjis, kimsib, kmjsib, kimcs,
                                   imjcs,
                                   kimis, imjis, id_2_span, sen_id)
    if span_type == 1:
        if abs(i - j) == 1:
            if dir == 0:
                heads[sen_id, i] = j
                heads_grands[sen_id, i] = k * sentence_length + j
            else:
                heads[sen_id, j] = i
                heads_grands[sen_id, j] = k * sentence_length + i
            return
        else:
            m = incomplete_backtrack[sen_id, span_id]
            if dir == 0:
                heads[sen_id, i] = j
                heads_grands[sen_id, i] = k * sentence_length + j
                left_span_id = jimis[span_id][m]
                right_span_id = kmjis[span_id][m]
                batch_backtracking(incomplete_backtrack, complete_backtrack, sibling_backtrack, left_span_id, 2,
                                   heads_grands,
                                   heads_siblings, heads, ijkss, jimcs, kmjcs, jimis, kmjis, kimsib, kmjsib, kimcs,
                                   imjcs,
                                   kimis, imjis, id_2_span, sen_id)
                batch_backtracking(incomplete_backtrack, complete_backtrack, sibling_backtrack, right_span_id, 1,
                                   heads_grands,
                                   heads_siblings, heads, ijkss, jimcs, kmjcs, jimis, kmjis, kimsib, kmjsib, kimcs,
                                   imjcs,
                                   kimis, imjis, id_2_span, sen_id)
            else:
                heads[sen_id, j] = i
                heads_grands[sen_id, j] = k * sentence_length + i
                left_span_id = kimis[span_id][m]
                right_span_id = imjis[span_id][m]
                batch_backtracking(incomplete_backtrack, complete_backtrack, sibling_backtrack, left_span_id, 1,
                                   heads_grands,
                                   heads_siblings, heads, ijkss, jimcs, kmjcs, jimis, kmjis, kimsib, kmjsib, kimcs,
                                   imjcs,
                                   kimis, imjis, id_2_span, sen_id)
                batch_backtracking(incomplete_backtrack, complete_backtrack, sibling_backtrack, right_span_id, 2,
                                   heads_grands,
                                   heads_siblings, heads, ijkss, jimcs, kmjcs, jimis, kmjis, kimsib, kmjsib, kimcs,
                                   imjcs,
                                   kimis, imjis, id_2_span, sen_id)

    if span_type == 2:
        m = sibling_backtrack[sen_id, span_id]
        if dir == 0:
            heads_siblings[sen_id, i] = k
        else:
            heads_siblings[sen_id, j] = k * sentence_length + i
        left_span_id = kimsib[span_id][m]
        right_span_id = kmjsib[span_id][m]
        batch_backtracking(incomplete_backtrack, complete_backtrack, sibling_backtrack, left_span_id, 0,
                           heads_grands,
                           heads_siblings, heads, ijkss, jimcs, kmjcs, jimis, kmjis, kimsib, kmjsib, kimcs,
                           imjcs,
                           kimis, imjis, id_2_span, sen_id)
        batch_backtracking(incomplete_backtrack, complete_backtrack, sibling_backtrack, right_span_id, 0,
                           heads_grands,
                           heads_siblings, heads, ijkss, jimcs, kmjcs, jimis, kmjis, kimsib, kmjsib, kimcs,
                           imjcs,
                           kimis, imjis, id_2_span, sen_id)
