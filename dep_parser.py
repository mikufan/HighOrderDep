import torch
import torch.autograd as autograd
from optparse import OptionParser
from parser_model import dep_model as HODP_MODEL
import utils
from tqdm import tqdm
import sys
import random
import numpy as np
import os
import pickle
import time

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="train", help="train file", metavar="FILE", default="data/toy_test")
    parser.add_option("--dev", dest="dev", help="dev file", metavar="FILE", default="data/wsj10_d")

    parser.add_option("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE")
    parser.add_option("--batch", type="int", dest="batchsize", default=1000)
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE",
                      default="output/neuralhighorder.model")
    parser.add_option("--wembedding", type="int", dest="wembedding_dim", default=100)
    parser.add_option("--pembedding", type="int", dest="pembedding_dim", default=25)
    parser.add_option("--hidden", type="int", dest="hidden_dim", default=100)
    parser.add_option("--nLayer", type="int", dest="n_layer", default=1)
    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")

    parser.add_option("--optim", type="string", dest="optim", default='adam')
    parser.add_option("--lr", type="float", dest="learning_rate", default=0.01)
    parser.add_option("--outdir", type="string", dest="output", default="output")
    parser.add_option("--sample_idx", type="int", dest="sample_idx", default=1000)
    parser.add_option("--feature_dim", type="int", dest="feature_dim", default=25)

    parser.add_option("--dropout", type="float", dest="dropout_ratio", default=0.25)
    parser.add_option("--order", type="int", dest="order", default=3)
    parser.add_option("--epochs", type="int", dest="epochs", default=50)
    parser.add_option("--do_eval", action="store_true", dest="do_eval", default=False)
    parser.add_option("--log", dest="log", help="log file", metavar="FILE", default="output/log")
    # parser.add_option("--sub_batch", dest="sub_batch_size", type="int", default=1000)
    parser.add_option("--length_filter", type="int", default=40)
    parser.add_option("--imbalanced_batch", action="store_true", default=False)
    parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)

    parser.add_option("--paramdec", dest="paramdec", help="Decoder parameters file", metavar="FILE",
                      default="paramdec.pickle")

    parser.add_option("--gpu", type="int", dest="gpu", default=-1, help='gpu id, set to -1 if use cpu mode')
    parser.add_option("--sparse_feature", action="store_true", default=False)
    parser.add_option("--combine_score", action="store_true", default=False)

    (options, args) = parser.parse_args()

    if options.gpu >= 0 and torch.cuda.is_available():
        print 'To use gpu' + str(options.gpu)


    def do_eval(dep_model, w2i, pos, options, feats=None, feature_type=None):
        print "===================================="
        print 'Do evaluation'
        if not options.sparse_feature:
            eval_sentences = utils.read_data(options.dev, True)
        else:
            eval_sentences = utils.read_sparse_data(options.dev, True, options.order)
        dep_model.eval()
        eval_data_list = utils.construct_parsing_data_list(eval_sentences, w2i, pos, options.length_filter,
                                                           options.sparse_feature, options.order, feature_type, feats)
        devpath = os.path.join(options.output, 'test_pred' + str(epoch + 1) + '_' + str(options.sample_idx))
        if not options.imbalanced_batch:
            eval_batch_data = utils.construct_batch_data(eval_data_list, options.batchsize)
        else:
            eval_batch_data = utils.construct_imbalanced_batch_data(eval_data_list,options.batchsize,options.order)

        for batch_id, one_batch in tqdm(enumerate(eval_batch_data), mininterval=2,
                                        desc=' -Tot it %d (epoch %d)' % (len(eval_batch_data), 0), leave=False,
                                        file=sys.stdout):
            if not options.sparse_feature:
                eval_batch_words, eval_batch_pos, eval_batch_parent, eval_batch_sen = [s[0] for s in one_batch], \
                                                                                      [s[1] for s in one_batch], \
                                                                                      [s[2] for s in one_batch], \
                                                                                      [s[3][0] for s in one_batch]
                eval_batch_words_v = torch.LongTensor(eval_batch_words)
                eval_batch_pos_v = torch.LongTensor(eval_batch_pos)
                eval_batch_parent_v = torch.LongTensor(eval_batch_parent)
                dep_model(eval_batch_words_v, eval_batch_pos_v, eval_batch_parent_v, eval_batch_sen)
            else:
                batch_parent = [s[2] for s in one_batch]
                batch_feats = [s[4] for s in one_batch]
                batch_sen = [s[3][0] for s in one_batch]
                batch_feats = torch.LongTensor(batch_feats)
                batch_parent_v = torch.LongTensor(batch_parent)
                dep_model.forward_sparse(batch_parent_v, batch_feats, batch_sen)
            if options.order == 1:
                del dep_model.hm_scores_table
            if options.order == 2:
                del dep_model.ghm_scores_table
                if options.combine_score:
                    del dep_model.hm_scores_table
            if options.order == 3:
                del dep_model.ghms_scores_table
            if options.gpu > -1 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        test_res = dep_model.parse_results
        utils.eval(test_res, eval_data_list, devpath, options.log + '_' + str(options.sample_idx), epoch)
        print "===================================="


    if not options.sparse_feature:
        w2i, pos, sentences = utils.read_data(options.train, False)
        features = None
        feature_type = None
    else:
        w2i, pos, features, sentences, feature_type = utils.read_sparse_data(options.train, False, options.order)
    print 'Data read'
    # torch.manual_seed(options.seed)

    data_list = utils.construct_parsing_data_list(sentences, w2i, pos, options.length_filter, options.sparse_feature,
                                                  options.order, feature_type, features)

    # batch_data = utils.construct_update_batch_data(data_list, options.batchsize)
    if options.imbalanced_batch:
        batch_data = utils.construct_imbalanced_batch_data(data_list, options.batchsize,options.order)
    else:
        batch_data = utils.construct_batch_data(data_list, options.batchsize)
    print 'Batch data constructed'
    high_order_dep_model = HODP_MODEL(w2i, pos, options, features)
    print 'Model constructed'
    if options.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(options.gpu)
        high_order_dep_model.cuda(options.gpu)

    for epoch in range(options.epochs):
        print 'Starting epoch', epoch
        high_order_dep_model.train()
        iter_loss = 0.0
        tot_batch = len(batch_data)
        random.shuffle(batch_data)
        for batch_id, one_batch in tqdm(enumerate(batch_data), mininterval=2,
                                        desc=' -Tot it %d (epoch %d)' % (tot_batch, 0), leave=False, file=sys.stdout):
            batch_loss_list = []
            if not options.sparse_feature:
                batch_words, batch_pos, batch_parent, batch_sen = [s[0] for s in one_batch], [s[1] for s in one_batch], [
                    s[2] for s in one_batch], [s[3][0] for s in one_batch]
                batch_words_v = torch.LongTensor(batch_words)
                batch_pos_v = torch.LongTensor(batch_pos)
                batch_parent_v = torch.LongTensor(batch_parent)
                batch_loss = high_order_dep_model(batch_words_v, batch_pos_v, batch_parent_v, batch_sen)
            else:
                batch_parent = [s[2] for s in one_batch]
                batch_feats = [s[4] for s in one_batch]
                batch_sen = [s[3][0] for s in one_batch]
                batch_feats = torch.LongTensor(batch_feats)
                batch_parent_v = torch.LongTensor(batch_parent)
                batch_loss = high_order_dep_model.forward_sparse(batch_parent_v, batch_feats, batch_sen)
            # start = time.clock()
            batch_loss.backward()
            high_order_dep_model.trainer.step()
            high_order_dep_model.trainer.zero_grad()
            # elasped = time.clock() - start
            # print "time cost in optimization " + str(elasped)
            if options.order == 1:
                del high_order_dep_model.hm_scores_table
            if options.order == 2:
                del high_order_dep_model.ghm_scores_table
                if options.combine_score:
                    del high_order_dep_model.hm_scores_table
            if options.order == 3:
                del high_order_dep_model.ghms_scores_table
            if options.gpu > -1 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            iter_loss += batch_loss.cpu()
        iter_loss /= tot_batch
        print ' loss for this iteration ', str(iter_loss.detach().data.numpy())
        if options.do_eval:
            if not options.sparse_feature:
                do_eval(high_order_dep_model, w2i, pos, options)
            else:
                do_eval(high_order_dep_model, w2i, pos, options, features, feature_type)

print 'Training finished'
