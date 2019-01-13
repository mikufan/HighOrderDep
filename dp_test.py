import eisner_layer_m1
import torch
import numpy as np

grand_score = torch.zeros((1, 5, 5, 5), dtype=torch.float)
sibling_score = torch.zeros((1, 5, 5, 5), dtype=torch.float)
grand_score.fill_(-np.inf)
grand_score[0, 0, 0, 1] = 0.1
grand_score[0, 0, 0, 2] = 0.4
grand_score[0, 0, 0, 3] = 0.1
grand_score[0, 0, 0, 4] = 0.1
grand_score[0, 0, 1, 2] = 0.1
grand_score[0, 0, 1, 3] = 0.1
grand_score[0, 0, 1, 4] = 0.1
grand_score[0, 0, 2, 1] = 0.4
grand_score[0, 1, 2, 3] = 0.1
grand_score[0, 1,]
