# CCNE: Collaborative Cross-Network Embedding Framework for Network Alignment
This repository contains the code of paper:  
 >H. -F. Zhang, G. Ren, X. Ding, L. Zhou and X. Zhang, "Collaborative Cross-Network Embedding Framework for Network Alignment," in IEEE Transactions on Network Science and Engineering, vol. 11, no. 3, pp. 2989-3001, May-June 2024, doi: 10.1109/TNSE.2024.3355479.

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

@ARTICLE{ren_ccne_2024,

  author={Zhang, Hai-Feng and Ren, Guojing and Ding, Xiao and Zhou, Li and Zhang, Xingyi},
  
  journal={IEEE Transactions on Network Science and Engineering}, 
  
  title={Collaborative Cross-Network Embedding Framework for Network Alignment}, 
  
  year={2024},
  
  volume={11},
  
  number={3},
  
  pages={2989-3001},
  
  doi={10.1109/TNSE.2024.3355479}
  
  }

  ## Baselines -updated on May 14, 2025
  CLF: <https://github.com/deepopo/CLF>
  
  PALE: `PALE_toy.py`
  
  IONE: <https://github.com/ColaLL/IONE>
  
  DeepLink: <https://github.com/KDD-HIEPT/DeepLink> or `DeepLink_toy.py`
  
  dNAME: `dNAME_toy.py`
  
  CAMU: `CAMU_toy.py`
  
  NeXtAlign: <https://github.com/sizhang92/NextAlign-KDD21>
  
  CENALP: <https://github.com/deepopo/CENALP>
  
  Grad-Align+: <https://github.com/jindeok/GradAlign_plus>
  
  More baselines: 
  
  <https://github.com/thanhtrunghuynh93/networkAlignment>
  
  <https://github.com/Allen517/alp-baselines>
