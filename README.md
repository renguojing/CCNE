# CCNE
Collaborative Cross-Network Embedding Framework for Network Alignment
### Author: Hai-Feng Zhang, Guo-Jing Ren, Xiao Ding, Li Zhou, and Xingyi Zhang

####Requirements
python                    3.8.12
gensim                    4.0.1
networkx                  2.6.3
numpy                     1.20.3
scikit-learn              0.24.2
torch                     1.9.0
torch-geometric           2.0.4

####Examples
If you want to run CCNE algorithm, use
`sh run_ccne.sh`
or
`python ccne.py --s_edge $data/douban/online/edgelist --t_edge data/douban/offline/raw/edgelist --gt_path ${PD}/anchor/node,split=0.8.test.dict --train_path ${PD}/anchor/node,split=0.8.train.dict`
