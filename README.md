# HighOrderDep
Running command for second-order dependency parsing:
python dep_parser.py --train data/WSJ_s2-21_tree_dep --dev data/WSJ_s23_tree_dep  --gpu 0 --batch 10 --order 2 --do_eval --epochs 50 --lr 0.001 