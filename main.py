from scipy import io
import math
import argparse
from time import *
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import warnings
import numpy as np

from model import SCDGN as Net
from dataset import *
from task import *
from utilis import *

from ipdb import set_trace

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    # import faulthandler
    # faulthandler.enable()
    parser = argparse.ArgumentParser(description='ICML')

    parser.add_argument('--dataname', type=str, default='cora', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=40, help='Training epochs.')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stop.')
    parser.add_argument('--train', type=int, default=1, help='Train or not.')
    
    # Dataset args
    parser.add_argument('--cut', type=str, default=False, help='The type of degree.')  ##
    parser.add_argument('--type', type=str,default='sys',help='sys or rw')    ##
    parser.add_argument('--knn', type=int, default=25, help='The K of KNN graph.')  ##
    parser.add_argument('--v_input', type=int, default=1, help='Degree of freedom of T distribution')  ##
    parser.add_argument('--sigma', type=float, default=0.5, help='Weight parameters for knn.')  ##
    
    # Optimizer args
    parser.add_argument('--imp_lr', type=float, default=1e-3, help='Learning rate of ICML.')  ##
    parser.add_argument('--exp_lr', type=float, default=1e-5, help='Learning rate of ICML.')  ##
    parser.add_argument('--imp_wd', type=float, default=1e-5, help='Weight decay of ICML.')
    parser.add_argument('--exp_wd', type=float, default=1e-5, help='Weight decay of ICML.')

    # GNN args
    parser.add_argument("--hid_dim", type=int, default=512, help='Hidden layer dim.')  ##
    parser.add_argument('--time', type=float, default=18, help='End time of ODE integrator.')  ##
    parser.add_argument('--method', type=str, default='dopri5', help="set the numerical solver: dopri5, euler, rk4, midpoint")
    parser.add_argument('--tol_scale', type=float, default=200, help='tol_scale .')  ##
    parser.add_argument('--add_source', type=str, default=True, help='Add source.')
    parser.add_argument('--dropout', type=float, default=0., help='drop rate.')  ##
    parser.add_argument('--n_layers', type=int, default=2, help='number of Linear.')  ##
    
    # Loss args
    parser.add_argument('--beta', type=float, default=1, help='Weight parameters for loss.')
    parser.add_argument('--gamma', type=float, default=1, help='Weight parameters for ICML.')
    
    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'
    set_seed(args.seed)
    print(args)
    begin_time = time()

    # Load data
    if args.dataname in ['amap', 'bat', 'eat', 'uat','corafull']:
        feat, label, A = load_data(args.dataname)
        labels = torch.from_numpy(label)
    else:
        data = io.loadmat('./data/{}.mat'.format(args.dataname))
        if args.dataname == 'wiki':
            feat = data['fea'].todense()
            A = data['W'].todense()
        elif args.dataname == 'pubmed':
            feat = data['fea']
            A = data['W'].todense()
        else:
            feat = data['fea']
            A = np.mat(data['W'])
        gnd = data['gnd'].T - 1
        labels = torch.from_numpy(gnd[0, :])
    
    feat = torch.from_numpy(feat).type(torch.FloatTensor)
    in_dim = feat.shape[1]
    args.N = N = feat.shape[0]
    norm_factor, edge_index, edge_weight, adj_norm, knn, Lap = cal_norm(A, args, feat)
    Lap_Neg = cal_Neg(adj_norm, knn, args)
    feat = feat.to(args.device)

    
    # Initial
    model = Net(N, edge_index, edge_weight, args).to(args.device)
    optimizer = optim.Adam([{'params':model.params_imp,'weight_decay':args.imp_wd, 'lr': args.imp_lr},
                            {'params':model.params_exp,'weight_decay':args.exp_wd, 'lr': args.exp_lr}])


    checkpt_file = './best/'+args.dataname+'_best.pt'
    print(checkpt_file)
    if args.train:
        cnt_wait = 0
        best_loss = 1e9
        best_epoch = 0
        best_acc = 0
        EYE = torch.eye(args.N).to(args.device)
        for epoch in range(1,args.epochs+1):
            model.train()
            optimizer.zero_grad()
            
            emb = model(knn, adj_norm, norm_factor)
            loss =( torch.trace(torch.mm(torch.mm(emb.t(), Lap), emb)) \
                        - args.beta*(torch.trace(torch.mm(torch.mm(emb.t(), Lap_Neg), emb))) \
                        + args.gamma*nn.MSELoss()(torch.mm(emb,emb.t()), EYE))/args.N 
            
            loss.backward()
            optimizer.step()
            
            if loss <= best_loss:
                best_loss = loss
                best_epoch = epoch
                cnt_wait = 0
                # acc, nmi, ari, f1 = clustering(emb.cpu().detach(), labels)
                # print(style.YELLOW + '\nClustering result: ACC:%1.2f  ||  NMI:%1.2f  ||  RI:%1.2f  ||  F-score:%1.2f '%(acc, nmi, ari, f1))
                torch.save(model.state_dict(), checkpt_file)
            else:
                cnt_wait += 1
            if cnt_wait == args.patience or math.isnan(loss):
                print('\nEarly stopping!', end='')
                break

            # print(style.MAGENTA + '\r\rEpoch={:03d}, loss={:.4f}'.format(epoch, loss.item()), end='  ')
            # print('')
        
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        emb = model(knn, adj_norm, norm_factor)
    
    # Clustering 
    emb = emb.cpu().detach().numpy()
    # TSNE_plot(emb, labels, args.dataname)
    clustering(emb, labels,args)
