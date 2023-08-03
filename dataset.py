import numpy as np
import torch
from sklearn.metrics import pairwise
import scipy
import scipy.sparse as sp
from torch_scatter import scatter_add
from torch_geometric.utils import to_undirected, to_scipy_sparse_matrix, degree, add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes

from ipdb import set_trace

def load_data(dataset_name, show_details=False):
    load_path = "./data/" + dataset_name + "/" + dataset_name
    feat = np.load(load_path+"_feat.npy", allow_pickle=True)
    label = np.load(load_path+"_label.npy", allow_pickle=True)
    adj = np.load(load_path+"_adj.npy", allow_pickle=True)
    if show_details:
        print("++++++++++++++++++++++++++++++")
        print("---details of graph dataset---")
        print("++++++++++++++++++++++++++++++")
        print("dataset name:   ", dataset_name)
        print("feature shape:  ", feat.shape)
        print("label shape:    ", label.shape)
        print("adj shape:      ", adj.shape)
        print("undirected edge num:   ", int(np.nonzero(adj)[0].shape[0]/2))
        print("category num:          ", max(label)-min(label)+1)
        print("category distribution: ")
        for i in range(max(label)+1):
            print("label", i, end=":")
            print(len(label[np.where(label == i)]))
        print("++++++++++++++++++++++++++++++")

    return feat, label, adj


def get_rw_adj(edge_index, norm_dim=1, fill_value=0., num_nodes=None, type='sys'):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    edge_weight = torch.ones((edge_index.size(1),), dtype=torch.float32, device=edge_index.device)
    
    if not fill_value == 0:
        edge_index, tmp_edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    indices = row if norm_dim == 0 else col
    deg = scatter_add(edge_weight, indices, dim=0, dim_size=num_nodes)
    # deg_inv_sqrt = deg.pow_(-1)
    # edge_weight = deg_inv_sqrt[indices] * edge_weight if norm_dim == 0 else edge_weight * deg_inv_sqrt[indices]
    
    if type=='sys':   
       deg_inv_sqrt = deg.pow_(-0.5)
       edge_weight = deg_inv_sqrt[indices] * edge_weight * deg_inv_sqrt[indices]
    else: 
       deg_inv_sqrt = deg.pow_(-1)
       edge_weight = deg_inv_sqrt[indices] * edge_weight if norm_dim == 0 else edge_weight * deg_inv_sqrt[indices]
    return edge_index, edge_weight

def adj_normalized(adj, type='sys'):
    row_sum = torch.sum(adj, dim=1)
    row_sum = (row_sum==0)*1+row_sum
    if type=='sys':
        d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return d_mat_inv_sqrt.mm(adj).mm(d_mat_inv_sqrt)
    else: 
        d_inv = torch.pow(row_sum, -1).flatten()
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat_inv = torch.diag(d_inv)
        return d_mat_inv.mm(adj)

def FeatureNormalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    rowsum = (rowsum == 0) * 1 + rowsum              #!!!!! 
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def compute_knn(args, features, distribution='t-distribution'):
    features = FeatureNormalize(features)
    # Dis = pairwise.euclidean_distances(self.data, self.data)
    # Dis = pairwise.manhattan_distances(self.data, self.data) 
    # Dis = pairwise.haversine_distances(self.data, self.data)
    Dis = pairwise.cosine_distances(features, features)
    Dis = Dis/np.max(np.max(Dis, 1))
    if distribution=='t-distribution':
        gamma = CalGamma(args.v_input)
        sim = gamma * np.sqrt(2 * np.pi) * np.power((1 + args.sigma*np.power(Dis,2) / args.v_input), -1 * (args.v_input + 1) / 2)
    else:
        sim = np.exp(-Dis/(args.sigma**2))

    K = args.knn
    if K>0:
        idx = sim.argsort()[:,::-1]
        sim_new = np.zeros_like(sim)
        for ii in range(0, len(sim_new)):
            sim_new[ii, idx[ii,0:K]] = sim[ii, idx[ii,0:K]]      
        Disknn = (sim_new + sim_new.T)/2
    else:
        Disknn = (sim + sim.T)/2
    
    Disknn = torch.from_numpy(Disknn).type(torch.FloatTensor)
    Disknn = torch.add(torch.eye(Disknn.shape[0]), Disknn)
    Disknn = adj_normalized(Disknn)

    return Disknn

def CalGamma(v):
    a = scipy.special.gamma((v + 1) / 2)
    b = np.sqrt(v * np.pi) * scipy.special.gamma(v / 2)
    out = a / b
    return out


# def cal_norm(edge_index0, args, feat=None, cut=False, num_nodes=None):
#     # calculate normalization factors: (2*D)^{-1/2} or (D)^{-1/2}
#     edge_index0 = sp.coo_matrix(edge_index0)
#     values = edge_index0.data  
#     indices = np.vstack((edge_index0.row, edge_index0.col))
#     edge_index0 = torch.LongTensor(indices).to(args.device) 
    
#     edge_weight = torch.ones((edge_index0.size(1),), dtype=torch.float32, device=args.device)
#     edge_index, _ = add_remaining_self_loops(edge_index0, edge_weight, 0, args.N)
    
#     if num_nodes is None:
#         num_nodes = edge_index.max()+1
#     D = degree(edge_index[0], num_nodes)  # 传入edge_index[0]计算节点出度, 该处为无向图，所以即计算节点度

#     if cut: 
#         D = torch.sqrt(1/D)
#         D[D == float("inf")] = 0.
#         edge_index = to_undirected(edge_index, num_nodes=num_nodes) 
#         row, col = edge_index
#         mask = row<col
#         edge_index = edge_index[:,mask]
#     else:
#         D = torch.sqrt(1/2/D)
#         D[D == float("inf")] = 0.
    
#     if D.dim() == 1:
#         D = D.unsqueeze(-1)
        
#     edge_index, edge_weight = get_rw_adj(edge_index, norm_dim=1, fill_value=1, num_nodes=args.N )
#     adj_norm = to_scipy_sparse_matrix(edge_index, edge_weight).todense()
#     adj_norm = torch.from_numpy(adj_norm).type(torch.FloatTensor).to(args.device)
#     Lap = 1./D - adj_norm
    
#     if feat == None:
#         return Lap
    
#     knn = compute_knn(args, feat).to(args.device)
#     feat = feat.to(args.device)

#     return D, edge_index, edge_weight, adj_norm, knn, Lap


def cal_norm(edge_index0, args, feat=None, cut=False, num_nodes=None):
    # calculate normalization factors: (2*D)^{-1/2} or (D)^{-1/2}
    edge_index0 = sp.coo_matrix(edge_index0)
    values = edge_index0.data  
    indices = np.vstack((edge_index0.row, edge_index0.col))
    edge_index0 = torch.LongTensor(indices).to(args.device) 
    
    edge_weight = torch.ones((edge_index0.size(1),), dtype=torch.float32, device=args.device)
    edge_index, _ = add_remaining_self_loops(edge_index0, edge_weight, 0, args.N)
    
    if num_nodes is None:
        num_nodes = edge_index.max()+1
    D = degree(edge_index[0], num_nodes)  # 传入edge_index[0]计算节点出度, 该处为无向图，所以即计算节点度

    if cut: 
        D = torch.sqrt(1/D)
        D[D == float("inf")] = 0.
        edge_index = to_undirected(edge_index, num_nodes=num_nodes) 
        row, col = edge_index
        mask = row<col
        edge_index = edge_index[:,mask]
    else:
        D = torch.sqrt(1/2/D)
        D[D == float("inf")] = 0.
    
    if D.dim() == 1:
        D = D.unsqueeze(-1)
        
    edge_index, edge_weight = get_rw_adj(edge_index, norm_dim=1, fill_value=1, num_nodes=args.N, type=args.type)
    adj_norm = to_scipy_sparse_matrix(edge_index, edge_weight).todense()
    adj_norm = torch.from_numpy(adj_norm).type(torch.FloatTensor).to(args.device)
    Lap = 1./D - adj_norm
    
    if feat == None:
        return Lap
    
    knn = compute_knn(args, feat).to(args.device)

    return D, edge_index, edge_weight, adj_norm, knn, Lap

def cal_Neg(knn, adj_norm, args):
    # Negative sample
    ones = torch.ones((args.N,args.N), dtype=torch.float32, device=args.device)
    zero = torch.zeros((args.N,args.N), dtype=torch.float32, device=args.device)
    Neg = torch.where((knn + adj_norm)==0, ones, zero).cpu()
    
    Lap_Neg = cal_norm(Neg, args)
    return Lap_Neg
     