# Model_list: MM_GNN
# Dataset_list: UGA50, GWU54, Northeastern19, Hamilton46, Caltech36, Howard90, UF21, Simmons81,  Flickr, Chameleon, Squirrel, Tulane29,
# mode_list: mlp, attention
# MMGNN
python3 main_social.py --model MM_GNN --num_layer 2 --repeat 10 --num_epoch 400 --gpu 0 --data_dir ../dataset --dataset Northeastern19 
# python3 main_social.py --model MM_GNN --num_layer 2 --repeat 10 --num_epoch 400 --gpu 0 --data_dir ../dataset --dataset Hamilton46 
# python3 main_social.py --model MM_GNN --num_layer 2 --repeat 10 --num_epoch 400 --gpu 0 --data_dir ../dataset --dataset UGA50 
# python3 main_social.py --model MM_GNN --num_layer 2 --repeat 10 --num_epoch 400 --gpu 0 --data_dir ../dataset --dataset GWU54 
# python3 main_social.py --model MM_GNN --num_layer 2 --repeat 10 --num_epoch 400 --gpu 0 --data_dir ../dataset --dataset Caltech36 
# python3 main_social.py --model MM_GNN --num_layer 2 --repeat 10 --num_epoch 400 --gpu 0 --data_dir ../dataset --dataset Howard90 
# python3 main_social.py --model MM_GNN --num_layer 2 --repeat 10 --num_epoch 400 --gpu 0 --data_dir ../dataset --dataset UF21 
# python3 main_social.py --model MM_GNN --num_layer 2 --repeat 10 --num_epoch 400 --gpu 0 --data_dir ../dataset --dataset Simmons81 
# python3 main_social.py --model MM_GNN --num_layer 2 --repeat 10 --num_epoch 400 --gpu 0 --data_dir ../dataset --dataset Tulane29 
