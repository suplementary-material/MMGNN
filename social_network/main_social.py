import torch
import torch_geometric
from torch_geometric.data import Data
import os.path as osp
import os
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Flickr
print(torch_geometric.__version__)
import numpy as np
import time
import sys
import logging
from torch.nn.utils import clip_grad_norm_
sys.path.append('./')
from dataset import Facebook100
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from utils import *



if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Training scripts for Higher Moment GNN model')
    ap.add_argument('--model', type=str, help='model name')
    ap.add_argument('--num_layer', type=int, default=2)
    ap.add_argument('--repeat', type=int, default=1)
    ap.add_argument('--num_epoch', type=int, default=200)
    ap.add_argument('--dataset', type=str, default='Cora',  help='dataset name')
    ap.add_argument('--data_dir', type=str, default='../data', help='path of dir of datasets')
    ap.add_argument('--gpu', type=str, default='0', help='id of gpu card to use')
    ap.add_argument('--running_id', type=str, default='0', help='experiment id for logging output')
    ap.add_argument('--log_dir', type=str, default=None, help='dir of log files, do not log if None')
    ap.add_argument('--hidden', type=int, default=16, help='fixed random seed by torch.manual_seed')
    ap.add_argument('--moment', type=int, default=1, help='max moment used in multi-moment model(MMGNN)')
    ap.add_argument('--mode', type=str, default='attention', help='mode to combine different moments feats, choose from mlp or attention')
    ap.add_argument('--seed', type=int, default=None, help='fixed random seed by torch.manual_seed')
    ap.add_argument('--auto_fixed_seed', action='store_true', default=True, help='fixed random seed of each run by run_id(0, 1, 2, ...)')
    ap.add_argument('--use_adj_norm', action='store_true', default=True, help='whether use adj normalization(row or symmetric norm).')
    ap.add_argument('--split_idx', type=int, default=0, help='split idx of multi-train/val/test mask dataset')
    ap.add_argument('--use_center_moment', action='store_true', default=False, help='whether to use center moment for MMGNN')
    ap.add_argument('--use_norm', action='store_true', default=False, help='whether to use layer norm for MMGNN')
    ap.add_argument('--lr', type=float, default=0.01, help='Initial learning rate for the optimizer.')
    ap.add_argument('--weight_decay', type=float, default=5e-3, help='weight decay for optimizer.')
    args = ap.parse_args()
    print(args)
    
    logger = None
    if args.log_dir is not None:
        if not osp.exists(args.log_dir):
            os.makedirs(args.log_dir)
        save_path = osp.join(args.log_dir, f'{args.dataset}_{args.model}_{args.num_layer}_{args.repeat}_{args.running_id}.log')
        logger = logging.getLogger(__name__)
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(save_path)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s]: %(message)s')
        handler.setFormatter(formatter)
        
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.addHandler(console)
    else:
        logging.basicConfig(level = logging.INFO,format = '[%(asctime)s]: %(message)s')
        logger = logging.getLogger(__name__)
    
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    print(device)
    

    final_acc = {
        'train': [],
        'val': [],
        'test': []
    }
    log_results = True
    for train_id in range(1, 1+args.repeat):
        logger.info('repeat {}/{}'.format(train_id, args.repeat))
        split_idx = args.split_idx
        if args.seed is not None:
            logger.info('Manual random seed:{}'.format(args.seed))
            data_split_seed = args.seed
            model_init_seed = train_id - 1
            set_random_seed(data_split_seed)
            logger.info('auto fixed data split seed to {}, model init seed to {}'.format(data_split_seed, model_init_seed))
        elif args.auto_fixed_seed:
            logger.info('auto fixed random seed to {}'.format(train_id-1))
            data_split_seed = int((train_id - 1) / 3)
            model_init_seed = int((train_id - 1) % 3)
            set_random_seed(data_split_seed)
            logger.info('auto fixed data split seed to {}, model init seed to {}'.format(data_split_seed, model_init_seed))

        # build datasets
        from torch_geometric.transforms import NormalizeFeatures, SIGN
        
        # transform = torch_geometric.transforms.Compose([SIGN(2)])
        transform = None
        
        dataset, data_split = build_dataset(args, transform=transform)
        # data_split=None
        data = dataset[0]
        
        # # save this data split
        # data_split = {
        #     'train': data.train_mask,
        #     'val': data.val_mask,
        #     'test': data.test_mask
        # }
        # path = os.path.join(args.data_dir, args.dataset)
        # torch.save(data_split, os.path.join(path, 'data_split.bin'))

        # split_idx = 1
        if args.dataset in ["Cornell", "Texas", "Wisconsin", "Actor", 'chameleon', 'squirrel']:
            data.train_mask = data.train_mask[:, split_idx]
            data.val_mask = data.val_mask[:, split_idx]
            data.test_mask = data.test_mask[:, split_idx]
        data = data.to(device)
        if data_split != None:
            data.train_mask = data_split['train'].to(device)
            data.val_mask = data_split['val'].to(device)
            data.test_mask = data_split['test'].to(device)

        print(data)
        logger.info('[Dataset-{}] train_num:{}, val_num:{}, test_num:{}, class_num:{}'.format(args.dataset, data.train_mask.sum().item(), data.val_mask.sum().item(), data.test_mask.sum().item(), dataset.num_classes))

        # torch.manual_seed(0) # 38

        # set_random_seed(seed_run)
        # print('model run seed:', seed_run)

        # build model
        model = build_model(args, dataset, device, model_init_seed)
        print(model)
        # build optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_acc = {
            'train': 0,
            'val': 0,
            'test': 0
        }

        # save_attn_fig = seed_run == 0
        save_attn_fig = False
        if save_attn_fig:
            atten_bucket_layer1 = [[] for _ in range(args.moment)]
            atten_bucket_layer2 = [[] for _ in range(args.moment)]

        for epoch in range(1, 1+args.num_epoch):
            t0 = time.time()
            # record attention value of mode attn-1 or attn-2
            if save_attn_fig:
                attn_score = F.softmax(model.convs[0].attn, dim=0)
                # attn_score = attn_score[:, :, :, 1].mean(-1).mean(-1).detach().cpu().numpy()
                attn_score = attn_score.mean(-1).mean(-1).mean(-1).detach().cpu().numpy()
                attn_score = np.around(attn_score, 3)
                for m in range(args.moment):
                    atten_bucket_layer1[m].append(attn_score[m])
                attn_score = F.softmax(model.convs[1].attn, dim=0)
                # attn_score = attn_score[:, :, :, 1].mean(-1).mean(-1).detach().cpu().numpy()
                attn_score = attn_score.mean(-1).mean(-1).mean(-1).detach().cpu().numpy()
                attn_score = np.around(attn_score, 3)
                for m in range(args.moment):
                    atten_bucket_layer2[m].append(attn_score[m])
                
            
            loss_train = train(data, model, optimizer)
            eval_res = test(data, model, dataset_name=args.dataset)
            log = 'Epoch: {:03d}, Loss:{:.4f} Train: {:.4f}, Val:{:.4f}, Test: {:.4f}, Time(s/epoch):{:.4f}'.format(epoch, loss_train, *eval_res, time.time() - t0)
            logger.info(log)
            if eval_res[1] > best_acc['val']:
                best_acc['train'] = eval_res[0]
                best_acc['val'] = eval_res[1]
                best_acc['test'] = eval_res[2]
            
        logger.info('[Run-{} score] {}'.format(train_id, best_acc))
        final_acc['train'].append(best_acc['train'])
        final_acc['val'].append(best_acc['val'])
        final_acc['test'].append(best_acc['test'])
        
        if save_attn_fig:
            linestyle_map = {
                0: '-',
                1: '--'
            }
            plt.figure()
            x = [1+i for i in range(args.num_epoch)]
            for m in range(args.moment):
                plt.plot(x, atten_bucket_layer1[m], label='attn-L{}-M{}'.format(1, m+1), linestyle=linestyle_map[m])
                plt.plot(x, atten_bucket_layer2[m], label='attn-L{}-M{}'.format(2, m+1), linestyle=linestyle_map[m])
            plt.legend(loc='best')
            plt.xlabel('epoch')
            plt.ylabel('attention')
            # print(atten_bucket_layer1, atten_bucket_layer2)
            plt.title('[{}:final attention]: L1:{:.2f}/{:.2f} | L2:{:.2f}/{:.2f}'.format(args.dataset, atten_bucket_layer1[0][-1], \
                atten_bucket_layer1[1][-1], atten_bucket_layer2[0][-1], atten_bucket_layer2[1][-1]))
            
            # file_dir = './visualization/moment_attention_analysis/init_m-reciprocal'
            # file_dir = './visualization/moment_attention_analysis/init_1-1_non_center'
            file_dir = './'
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            fig_save_path = os.path.join(file_dir, 'moment_attn_{}_{}.png'.format(args.dataset, args.mode))
            plt.savefig(fig_save_path)
    best_test_run  = np.argmax(final_acc['test'])
    final_acc_avg = {}
    final_acc_std = {}
    for key in final_acc:
        best_acc[key] = max(final_acc[key])
        final_acc_avg[key] = torch.topk(torch.Tensor(final_acc[key]), k=5).values.mean().item()
        final_acc_std[key] = np.std(final_acc[key])
    logger.info('[Average Score] {} '.format(final_acc_avg))
    logger.info('[std Score] {} '.format(final_acc_std))
    logger.info('[Best Score] {}'.format(best_acc))
    logger.info('[Best test run] {}'.format(best_test_run))
    if log_results:
        f_path = f'../res/MMGNN_optimize_{args.dataset}.csv'
        with open(f_path, 'a+') as fw:
            # fw.write('Dataset,Model,use_adj_norm,use_center_moment,moment,mode,avg_test_acc,exp_acc_list\n')
            avg_test_acc = final_acc_avg['test']
            test_acc_list = str(final_acc['test']).replace(',', ';').replace('[', '').replace(']', '').replace(' ', '')
            fw.write(f'{args.dataset} | lr={args.lr} | wd={args.weight_decay} | hidden={args.hidden},{args.model},{args.use_adj_norm},{args.use_center_moment},{args.moment},{args.mode},{avg_test_acc},{test_acc_list}\n')
    print(args)
    



