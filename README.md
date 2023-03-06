# CCNE: Collaborative Cross-Network Embedding Framework for Network Alignment
This repository contains the code of paper:  
 >Zhang, Hai-Feng and Ren, Guo-Jing and Ding, Xiao and Zhou, Li and Zhang, Xingyi. (2022). Collaborative Cross-Network Embedding Framework for Network Alignment.

#### Requirements
python                    3.8.12

gensim                    4.0.1

networkx                  2.6.3

numpy                     1.20.3

scikit-learn              0.24.2

torch                     1.9.0

torch-geometric           2.0.4

#### Examples
If you want to run CCNE algorithm on Douban online-offline dataset with training ratio 0.8, run the following command in the home directory of this project:
`sh run_ccne.sh`
or
`python ccne.py --s_edge data/douban/online/edgelist --t_edge data/douban/offline/raw/edgelist --gt_path data/douban/anchor/node,split=0.8.test.dict --train_path data/douban/anchor/node,split=0.8.train.dict`

## Reference  
If you are interested in our researches, please cite our papers:  

	@article{zhang2022collaborative,
	title = {Collaborative Cross-Network Embedding Framework for Network Alignment},
	author = {Zhang, Hai-Feng and Ren, Guo-Jing and Ding, Xiao and Zhou, Li and Zhang, Xingyi},
	year = 2022
	}
