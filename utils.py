import re
import random
from collections import Counter
from itertools import groupby
import numpy as np


class ConllEntry:
    def __init__(self, id, form, lemma, cpos, pos, feats=None, parent_id=None, relation=None, deps=None, misc=None):
        self.id = id
        self.form = form
        self.norm = normalize(form)
        self.cpos = cpos.upper()
        self.pos = pos.upper()
        self.parent_id = parent_id
        self.relation = relation

        self.lemma = lemma
        self.feats = feats
        self.deps = deps
        self.misc = misc

        # self.pred_parent_id = None
        # self.pred_relation = None

    def __str__(self):
        values = [str(self.id), self.form, self.lemma, self.cpos, self.pos, self.feats,
                  str(self.parent_id) if self.parent_id is not None else None, self.relation, self.deps,
                  self.misc]
        return '\t'.join(['_' if v is None else v for v in values])


numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");


def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()


def memoize(func):
    mem = {}

    def helper(*args, **kwargs):
        key = (args, frozenset(kwargs.items()))
        if key not in mem:
            mem[key] = func(*args, **kwargs)
        return mem[key]

    return helper


def construct_update_batch_data(data_list, batch_size):
    random.shuffle(data_list)
    batch_data = []
    len_datas = len(data_list)
    num_batch = len_datas // batch_size
    if not len_datas % batch_size == 0:
        num_batch += 1
    for i in range(num_batch):
        start_idx = i * batch_size
        end_idx = min(len_datas, (i + 1) * batch_size)
        batch_data.append(data_list[start_idx:end_idx])
    return batch_data


@memoize
def constituent_index(sentence_length, multiroot):
    counter_id = 0
    basic_span = []
    id_2_span = {}
    for left_idx in range(sentence_length):
        for right_idx in range(left_idx, sentence_length):
            for dir in range(2):
                id_2_span[counter_id] = (left_idx, right_idx, dir)
                counter_id += 1

    span_2_id = {s: id for id, s in id_2_span.items()}

    for i in range(sentence_length):
        if i != 0:
            id = span_2_id.get((i, i, 0))
            basic_span.append(id)
        id = span_2_id.get((i, i, 1))
        basic_span.append(id)

    ijss = []
    ikcs = [[] for _ in range(counter_id)]
    ikis = [[] for _ in range(counter_id)]
    kjcs = [[] for _ in range(counter_id)]
    kjis = [[] for _ in range(counter_id)]

    for l in range(1, sentence_length):
        for i in range(sentence_length - l):
            j = i + l
            for dir in range(2):
                ids = span_2_id[(i, j, dir)]
                for k in range(i, j + 1):
                    if dir == 0:
                        if k < j:
                            # two complete spans to form an incomplete span
                            idli = span_2_id[(i, k, dir + 1)]
                            ikis[ids].append(idli)
                            idri = span_2_id[(k + 1, j, dir)]
                            kjis[ids].append(idri)
                            # one complete span,one incomplete span to form a complete span
                            idlc = span_2_id[(i, k, dir)]
                            ikcs[ids].append(idlc)
                            idrc = span_2_id[(k, j, dir)]
                            kjcs[ids].append(idrc)

                    else:
                        if k < j and ((not (i == 0 and k != 0) and not multiroot) or multiroot):
                            # two complete spans to form an incomplete span
                            idli = span_2_id[(i, k, dir)]
                            ikis[ids].append(idli)
                            idri = span_2_id[(k + 1, j, dir - 1)]
                            kjis[ids].append(idri)
                        if k > i:
                            # one incomplete span,one complete span to form a complete span
                            idlc = span_2_id[(i, k, dir)]
                            ikcs[ids].append(idlc)
                            idrc = span_2_id[(k, j, dir)]
                            kjcs[ids].append(idrc)

                ijss.append(ids)

    return span_2_id, id_2_span, ijss, ikcs, ikis, kjcs, kjis, basic_span


class data_sentence:
    def __init__(self, id, entry_list):
        self.id = id
        self.entries = entry_list
        self.size = len(entry_list)

    def set_data_list(self, words, pos):
        word_list = list()
        pos_list = list()
        for entry in self.entries:
            if entry.norm in words.keys():
                word_list.append(words[entry.norm])
            else:
                word_list.append(words['<UNKNOWN>'])
            if entry.pos in pos.keys():
                pos_list.append(pos[entry.pos])
            else:
                pos_list.append(pos['<UNKNOWN-POS>'])
        return word_list, pos_list

    def set_parsing_data_list(self, words, pos):
        word_list = list()
        pos_list = list()
        parent_list = list()
        for entry in self.entries:
            if words.get(entry.norm) is not None:
                word_list.append(words[entry.norm])
            else:
                word_list.append(words['<UNKNOWN>'])
            if pos.get(entry.pos) is not None:
                pos_list.append(pos[entry.pos])
            else:
                pos_list.append(pos['<UNKNOWN-POS>'])
            parent_list.append(entry.parent_id)
        return word_list, pos_list, parent_list

    def __str__(self):
        return '\t'.join([e for e in self.entries])


def read_conll(fh):
    root = ConllEntry(0, '*root*', '*root*', 'ROOT-CPOS', 'ROOT-POS', '_', -1, 'rroot', '_', '_')
    tokens = [root]
    for line in fh:
        tok = line.strip().split('\t')
        if not tok or line.strip() == '':
            if len(tokens) > 1: yield tokens
            tokens = [root]
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                tokens.append(line.strip())
            else:
                # if tok[3][0] == 'V':
                #    tok[3] = "V"
                tokens.append(ConllEntry(int(tok[0]), tok[1], tok[2], tok[4], tok[3], tok[5],
                                         int(tok[6]) if tok[6] != '_' else -1, tok[7], tok[8], tok[9]))
    if len(tokens) > 1:
        yield tokens


def read_data(conll_path, isPredict):
    sentences = []
    if not isPredict:
        wordsCount = Counter()
        posCount = Counter()
        s_counter = 0
        with open(conll_path, 'r') as conllFP:
            for sentence in read_conll(conllFP):
                wordsCount.update([node.norm for node in sentence if isinstance(node, ConllEntry)])
                posCount.update([node.pos for node in sentence if isinstance(node, ConllEntry)])
                ds = data_sentence(s_counter, sentence)
                sentences.append(ds)
                s_counter += 1
        wordsCount['<UNKNOWN>'] = 0
        # posCount['<UNKNOWN-POS>'] = 0
        return {w: i for i, w in enumerate(wordsCount.keys())}, {p: i for i, p in enumerate(
            posCount.keys())}, sentences
    else:
        with open(conll_path, 'r') as conllFP:
            s_counter = 0
            for sentence in read_conll(conllFP):
                ds = data_sentence(s_counter, sentence)
                sentences.append(ds)
                s_counter += 1
        return sentences


def construct_batch_data(data_list, batch_size):
    data_list.sort(key=lambda x: len(x[0]))
    grouped = [list(g) for k, g in groupby(data_list, lambda s: len(s[0]))]
    batch_data = []
    for group in grouped:
        sub_batch_data = get_batch_data(group, batch_size)
        batch_data.extend(sub_batch_data)
    return batch_data


def get_batch_data(grouped_data, batch_size):
    batch_data = []
    len_datas = len(grouped_data)
    num_batch = len_datas // batch_size
    if not len_datas % batch_size == 0:
        num_batch += 1

    for i in range(num_batch):
        start_idx = i * batch_size
        end_idx = min(len_datas, (i + 1) * batch_size)
        batch_data.append(grouped_data[start_idx:end_idx])
    return batch_data


def get_index(b, id):
    id_a = id // b
    id_b = id % b
    return (id_a, id_b)


def eval(predicted, gold, test_path, log_path, epoch):
    correct_counter = 0
    total_counter = 0
    for s in range(len(gold)):
        ps = predicted[s][0]
        gs = gold[s]
        for i, e in enumerate(gs.entries):
            if i == 0:
                continue
            if ps[i] == e.parent_id:
                correct_counter += 1
            total_counter += 1
    accuracy = float(correct_counter) / total_counter
    print 'UAS is ' + str(accuracy * 100) + '%'
    f_w = open(test_path, 'w')
    for s, sentence in enumerate(gold):
        for entry in sentence.entries:
            f_w.write(str(entry.norm) + ' ')
        f_w.write('\n')
        for entry in sentence.entries:
            f_w.write(str(entry.pos) + ' ')
        f_w.write('\n')
        for i in range(len(sentence.entries)):
            f_w.write(str(sentence.entries[i].parent_id) + ' ')
        f_w.write('\n')
        for i in range(len(sentence.entries)):
            f_w.write(str(int(predicted[s][1][i])) + ' ')
        f_w.write('\n')
        for i in range(len(sentence.entries)):
            f_w.write(str(int(predicted[s][0][i])) + ' ')
        f_w.write('\n')
        f_w.write('\n')
    f_w.close()
    if epoch == 0:
        log = open(log_path, 'w')
        # log.write("UAS for epoch " + str(epoch))
        # log.write('\n')
        # log.write('\n')
        log.write(str(accuracy))
        log.write('\n')
        log.write('\n')
    else:
        log = open(log_path, 'a')
        # log.write("UAS for epoch " + str(epoch))
        # log.write('\n')
        # log.write('\n')
        log.write(str(accuracy))
        log.write('\n')
        log.write('\n')


def write_distribution(dmv_model):
    path = "output/dis_log"
    lex_path = "output/lex_log"
    for t in range(dmv_model.tag_num):
        log_path = path + str(t)
        head_idx = dmv_model.pos["VB"]
        writer = open(log_path, 'w')
        dist = dmv_model.trans_param[head_idx, :, t, :, :, :]
        for c in range(len(dmv_model.pos)):
            for ct in range(dmv_model.tag_num):
                for dir in range(2):
                    for cv in range(dmv_model.cvalency):
                        writer.write(str(dist[c, ct, dir, cv]))
                        writer.write('\n')
        if dmv_model.tag_num > 1:
            lex_log_path = lex_path + str(t)
            lex_writer = open(lex_log_path, 'w')
            lex_dist = dmv_model.lex_param[head_idx, t, :]
            min = np.min(np.array(lex_dist))
            for w in range(len(dmv_model.vocab)):
                if lex_dist[w] > min:
                    lex_writer.write(str(lex_dist[w]))
                    lex_writer.write('\n')


def construct_data_list(sentences, words, pos):
    data_list = list()
    sen_idx = 0
    for s in sentences:
        s_word, s_pos = s.set_data_list(words, pos)
        s_data_list = list()
        s_data_list.append(s_word)
        s_data_list.append(s_pos)
        s_data_list.append([sen_idx])
        data_list.append(s_data_list)
        sen_idx += 1
    return data_list


def construct_parsing_data_list(sentences, words, pos):
    data_list = list()
    sen_idx = 0
    for s in sentences:
        s_word, s_pos, s_parent = s.set_parsing_data_list(words, pos)
        s_data_list = list()
        s_data_list.append(s_word)
        s_data_list.append(s_pos)
        s_data_list.append(s_parent)
        s_data_list.append([sen_idx])
        data_list.append(s_data_list)
        sen_idx += 1
    return data_list


@memoize
def m1_constituent_index(sentence_length, multiroot):
    counter_id = 0
    basic_span = []
    id_2_span = {}
    id_2_sib = {}
    view_spans = []
    for grand_idx in range(sentence_length):
        for left_idx in range(sentence_length):
            for right_idx in range(left_idx, sentence_length):
                sibling_candidates = [s for s in range(left_idx + 1, right_idx + 1)]
                for dir in range(2):
                    if grand_idx < left_idx or grand_idx > right_idx or (
                                        grand_idx == left_idx and left_idx == 0 and dir == 1):
                        id_2_span[counter_id] = (grand_idx, left_idx, right_idx, dir)
                        if dir == 0:
                            id_2_sib[counter_id] = sibling_candidates[0:len(sibling_candidates) - 1]
                        else:
                            id_2_sib[counter_id] = sibling_candidates[1:]
                        counter_id += 1

    span_2_id = {s: id for id, s in id_2_span.items()}
    basic_span.append(span_2_id.get((0, 0, 0, 1)))
    for i in range(sentence_length):
        for j in range(sentence_length):
            if j != 0 and i != j:
                id = span_2_id.get((i, j, j, 0))
                basic_span.append(id)
                id = span_2_id.get((i, j, j, 1))
                basic_span.append(id)

    ijkss = []
    jimcs = [[] for _ in range(counter_id)]
    kmjcs = [[] for _ in range(counter_id)]
    jimis = [[] for _ in range(counter_id)]
    kmjis = [[] for _ in range(counter_id)]
    jimics = [[] for _ in range(counter_id)]
    kmjics = [[] for _ in range(counter_id)]
    kimsib = [[] for _ in range(counter_id)]
    kmjsib = [[] for _ in range(counter_id)]
    kimcs = [[] for _ in range(counter_id)]
    imjcs = [[] for _ in range(counter_id)]
    kimis = [[] for _ in range(counter_id)]
    imjis = [[] for _ in range(counter_id)]
    kimics = [[] for _ in range(counter_id)]
    imjics = [[] for _ in range(counter_id)]

    for l in range(1, sentence_length):
        for i in range(sentence_length - l):
            j = i + l
            for k in range(sentence_length):
                for dir in range(2):
                    if k < i or k > j or (k == i and i == 0 and dir == 1):
                        ids = span_2_id[(k, i, j, dir)]
                        for m in range(i, j + 1):
                            if dir == 0:
                                if m < j:
                                    # two complete spans to form a sibling span
                                    idlc = span_2_id[(k, i, m, dir + 1)]
                                    kimsib[ids].append(idlc)
                                    idrc = span_2_id[(k, m + 1, j, dir)]
                                    kmjsib[ids].append(idrc)

                                    # one complete spans,one incomplete span form a complete span
                                    idlc = span_2_id[(j, i, m, dir)]
                                    jimcs[ids].append(idlc)
                                    idri = span_2_id[(k, m, j, dir)]
                                    kmjcs[ids].append(idri)
                                    if m > i and m < j:
                                        # one incomplete span,one sibling span to form an incomplete span
                                        idlsib = span_2_id[(j, i, m, dir)]
                                        jimis[ids].append(idlsib)
                                        idri = span_2_id[(k, m, j, dir)]
                                        kmjis[ids].append(idri)
                                    if m == j - 1:
                                        # two complete spans to form an incomplete span
                                        idlc = span_2_id[(j, i, m, dir + 1)]
                                        jimics[ids].append(idlc)
                                        idrc = span_2_id[(k, m + 1, j, dir)]
                                        kmjics[ids].append(idrc)
                            else:
                                if m < j:
                                    # two complete spans to form a sibling span
                                    idlc = span_2_id[(k, i, m, dir)]
                                    kimsib[ids].append(idlc)
                                    idrc = span_2_id[(k, m + 1, j, dir - 1)]
                                    kmjsib[ids].append(idrc)
                                if m > i:
                                    # one complete span and one incomplete span to form a complete span
                                    idli = span_2_id[(k, i, m, dir)]
                                    kimcs[ids].append(idli)
                                    idrc = span_2_id[(i, m, j, dir)]
                                    imjcs[ids].append(idrc)
                                if m < j and m > i:
                                    # one incomplete span,one sibling span to form a incomplete span
                                    idli = span_2_id[(k, i, m, dir)]
                                    kimis[ids].append(idli)
                                    idrsib = span_2_id[(i, m, j, dir)]
                                    imjis[ids].append(idrsib)
                                elif m == i:
                                    # two complete spans to form an incomplete span
                                    idlc = span_2_id[(k, i, m, dir)]
                                    kimics[ids].append(idlc)
                                    idrc = span_2_id[(i, m + 1, j, dir - 1)]
                                    imjics[ids].append(idrc)

                        ijkss.append(ids)
                        view_spans.append((k, i, j, dir))

    return span_2_id, id_2_span, ijkss, jimcs, kmjcs, jimis, kmjis, jimics, kmjics, kimsib, kmjsib, \
           kimcs, imjcs, kimis, imjis, kimics, imjics, basic_span, id_2_sib
