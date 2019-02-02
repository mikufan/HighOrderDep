import eisner_layer_m1 as EL
import torch
import numpy as np

g_heads_grand_sibling = torch.LongTensor([[0, 112, 23, 24, 0]])
ghms_score = torch.zeros((1, 5, 5, 5, 5), dtype=torch.float)
ghms_score.fill_(-np.inf)
ghms_score[0, 1, 0, 0, 0] = 0.1
ghms_score[0, 1, 0, 2, 2] = 0.1
ghms_score[0, 1, 0, 3, 2] = 0.1
ghms_score[0, 1, 0, 3, 3] = 0.1
ghms_score[0, 1, 0, 4, 2] = 0.1
ghms_score[0, 1, 0, 4, 3] = 0.1
ghms_score[0, 1, 0, 4, 4] = 0.1
ghms_score[0, 1, 3, 2, 2] = 0.1
ghms_score[0, 1, 4, 2, 2] = 5
ghms_score[0, 1, 4, 3, 2] = 0.1
ghms_score[0, 1, 4, 3, 3] = 0.1
ghms_score[0, 2, 0, 0, 0] = 0.1
ghms_score[0, 2, 0, 1, 1] = 0.1
ghms_score[0, 2, 0, 3, 3] = 0.1
ghms_score[0, 2, 0, 4, 3] = 5
ghms_score[0, 2, 0, 4, 4] = 0.1
ghms_score[0, 2, 1, 3, 3] = 0.1
ghms_score[0, 2, 1, 4, 3] = 0.1
ghms_score[0, 2, 1, 4, 4] = 0.1
ghms_score[0, 2, 3, 1, 1] = 0.1
ghms_score[0, 2, 4, 1, 1] = 0.1
ghms_score[0, 2, 4, 3, 3] = 0.1
ghms_score[0, 3, 0, 0, 0] = 0.1
ghms_score[0, 3, 0, 1, 1] = 0.1
ghms_score[0, 3, 0, 1, 2] = 0.1
ghms_score[0, 3, 0, 2, 2] = 0.1
ghms_score[0, 3, 0, 4, 4] = 5
ghms_score[0, 3, 1, 2, 2] = 0.1
ghms_score[0, 3, 1, 4, 4] = 0.1
ghms_score[0, 3, 2, 4, 4] = 0.1
ghms_score[0, 3, 4, 1, 1] = 0.1
ghms_score[0, 3, 4, 1, 2] = 0.1
ghms_score[0, 3, 4, 2, 2] = 0.1
ghms_score[0, 4, 0, 0, 0] = 5
ghms_score[0, 4, 0, 1, 1] = 0.1
ghms_score[0, 4, 0, 1, 2] = 0.1
ghms_score[0, 4, 0, 1, 3] = 0.1
ghms_score[0, 4, 0, 2, 2] = 0.1
ghms_score[0, 4, 0, 2, 3] = 0.1
ghms_score[0, 4, 0, 3, 3] = 0.1
ghms_score[0, 4, 1, 2, 2] = 0.1
ghms_score[0, 4, 1, 2, 3] = 0.1
ghms_score[0, 4, 1, 3, 3] = 0.1
ghms_score[0, 4, 2, 3, 3] = 0.1
heads_grand_sibling, heads = EL.batch_parse(ghms_score, g_heads_grand_sibling)
print heads_grand_sibling
print heads
