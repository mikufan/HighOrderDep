import eisner_layer_m1 as EL
import torch
import numpy as np

ghms_score = torch.zeros((1, 4, 4, 4, 4), dtype=torch.float)
ghms_score.fill_(0.1)
ghms_score[0,1,0,0,0]+=1
ghms_score[0,2,0,1,1]+=1
ghms_score[0,3,0,1,2]+=1


heads_grand_sibling, heads,final_score = EL.batch_parse(ghms_score)
print (heads_grand_sibling)
print (heads)
print (final_score)
