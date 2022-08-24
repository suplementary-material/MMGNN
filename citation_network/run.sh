python3 main_citation.py --data cora --test --num_epoch 400 --hidden 64
python3 main_citation.py --data citeseer --hidden 256  --num_epoch 600
python3 main_citation.py --data pubmed --hidden 256 --lamda 0.4 --wd1 5e-4 --num_epoch 600
