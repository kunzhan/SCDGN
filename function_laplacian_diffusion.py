import torch
from torch import nn
import torch_sparse
from torch_geometric.nn.conv import MessagePassing
from ipdb import set_trace
  
class ODEFunc(MessagePassing):
    # currently requires in_features = out_features
    def __init__(self, opt, data, device):
        super(ODEFunc, self).__init__()
        self.opt = opt
        self.device = device
        self.edge_index = None
        self.edge_weight = None
        self.attention_weights = None
        self.alpha_train = nn.Parameter(torch.tensor(0.0))
        self.beta_train = nn.Parameter(torch.tensor(0.0))
        self.x0 = None
        self.nfe = 0
        self.alpha_sc = nn.Parameter(torch.ones(1))
        self.beta_sc = nn.Parameter(torch.ones(1))

    def __repr__(self):
        return self.__class__.__name__
  
# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class LaplacianODEFunc(ODEFunc):
    # currently requires in_features = out_features
    def __init__(self, args, data, device):
        super(LaplacianODEFunc, self).__init__(args, data, device)
        self.args = args
    
    def forward(self, t, x):  # the t param is needed by the ODE solver.
        self.nfe += 1
        ax = torch_sparse.spmm(self.edge_index, self.edge_weight, x.shape[0], x.shape[0], x)
        
        alpha = torch.sigmoid(self.alpha_train)
        
        f = alpha * (ax - x)
        if self.args.add_source:
            f = f + self.beta_train * self.x0
        
        return f
