
import itertools
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


def JSE_loss(zs, zs_n=None, batch=None, sigma=None, neg_by_crpt=False, **kwargs):
    '''The Jensen-Shannon Estimator of Mutual Information used in contrastive learning. The
    implementation follows the paper `Learning deep representations by mutual information 
    estimation and maximization <https://arxiv.org/abs/1808.06670>`_.
    
    .. note::
        The JSE loss implementation can produce negative values because a :obj:`-2log2` shift is 
        added to the computation of JSE, for the sake of consistency with other f-convergence 
        losses.
    
    Args:
        zs (list, optional): List of tensors of shape [batch_size, z_dim].
        zs_n (list, optional): List of tensors of shape [nodes, z_dim].
        batch (Tensor, optional): Required when both :obj:`zs` and :obj:`zs_n` are given.
        sigma (ndarray, optional): A 2D-array of shape [n_views, n_views] with boolean values, 
            indicating contrast between which two views are computed. Only required 
            when number of views is greater than 2. If :obj:`sigma[i][j]` = :obj:`True`, 
            JSE between :math:`view_i` and :math:`view_j` will be computed.
        neg_by_crpt (bool, optional): The mode to obtain negative samples in JSE. If True, 
            obtain negative samples by performing corruption. Otherwise, consider pairs of
            different graph samples as negative pairs.

    :rtype: :class:`Tensor`
    '''
    if zs_n is not None:
        assert len(zs_n) == len(zs)
        assert batch is not None
        
        jse = (JSE_local_global_negative_paired 
               if neg_by_crpt else JSE_local_global)
        
        if len(zs) == 1:
            return jse(zs[0], zs_n[0], batch)
        elif len(zs) == 2:
            return (jse(zs[0], zs_n[1], batch) +
                    jse(zs[1], zs_n[0], batch))
        else:
            assert len(zs) == len(sigma)
            loss = 0
            for (i, j) in itertools.combinations(range(len(zs)), 2):
                if sigma[i][j]:
                    loss += (jse(zs[i], zs_n[j], batch) +
                             jse(zs[j], zs_n[i], batch))
            return loss

    else:
        jse = JSE_global_global
        if len(zs) == 2:
            return jse(zs[0], zs[1])
        elif len(zs) > 2:
            assert len(zs) == len(sigma)
            loss = 0
            for (i, j) in itertools.combinations(range(len(zs)), 2):
                if sigma[i][j]:
                    loss += jse(zs[i], zs[j])
            return loss



def JSE_local_global_negative_paired(z_g, z_n, batch):
    '''
    Args:
        z_g: of size [2*n_batch, dim]
        z_n: of size [2*n_batch*nodes_per_batch, dim]
    '''
    device = z_g.device
    num_graphs = int(z_g.shape[0]/2)  # 4
    num_nodes = int(z_n.shape[0]/2) # 4*2000
    z_g, _ = torch.split(z_g, num_graphs)
    z_n, z_n_crpt = torch.split(z_n, num_nodes)

    num_sample_nodes = int(num_nodes / num_graphs)
    z_n = torch.split(z_n, num_sample_nodes)
    z_n_crpt = torch.split(z_n_crpt, num_sample_nodes)

    d_pos = torch.cat([torch.matmul(z_g[i], z_n[i].t()) for i in range(num_graphs)])  # [1, 8000]
    d_neg = torch.cat([torch.matmul(z_g[i], z_n_crpt[i].t()) for i in range(num_graphs)])  # [1, 8000]
        
    logit = torch.unsqueeze(torch.cat((d_pos, d_neg)), 0)  # [1, 16000]
    lb_pos = torch.ones((1, num_nodes)).to(device)  # [1, 8000]
    lb_neg = torch.zeros((1, num_nodes)).to(device)  # [1, 8000]
    lb = torch.cat((lb_pos, lb_neg), 1)

    b_xent = nn.BCEWithLogitsLoss()
    loss = b_xent(logit, lb) * 0.5 # following mvgrl-node
    return loss
    

def JSE_local_global(z_g, z_n, batch):
    '''
    Args:
        z_g: Tensor of shape [n_graphs, z_dim].
        z_n: Tensor of shape [n_nodes, z_dim].
        batch: Tensor of shape [n_graphs].
    '''
    device = z_g.device
    num_graphs = z_g.shape[0]
    num_nodes = z_n.shape[0]

    pos_mask = torch.zeros((num_nodes, num_graphs)).to(device)
    neg_mask = torch.ones((num_nodes, num_graphs)).to(device)
    for nodeidx, graphidx in enumerate(batch):
        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.

    d_prime = torch.matmul(z_n, z_g.t())

    E_pos = get_expectation(d_prime * pos_mask, positive=True).sum()
    E_pos = E_pos / num_nodes
    E_neg = get_expectation(d_prime * neg_mask, positive=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))
    return E_neg - E_pos


def JSE_global_global(z1, z2):
    '''
    Args:
        z1, z2: Tensor of shape [batch_size, z_dim].
    '''
    device = z1.device
    num_graphs = z1.shape[0]

    pos_mask = torch.zeros((num_graphs, num_graphs)).to(device)
    neg_mask = torch.ones((num_graphs, num_graphs)).to(device)
    for graphidx in range(num_graphs):
        pos_mask[graphidx][graphidx] = 1.
        neg_mask[graphidx][graphidx] = 0.

    d_prime = torch.matmul(z1, z2.t())

    E_pos = get_expectation(d_prime * pos_mask, positive=True).sum()
    E_pos = E_pos / num_graphs
    E_neg = get_expectation(d_prime * neg_mask, positive=False).sum()
    E_neg = E_neg / (num_graphs * (num_graphs - 1))
    return E_neg - E_pos


def get_expectation(masked_d_prime, positive=True):
    '''
    Args:
        masked_d_prime: Tensor of shape [n_graphs, n_graphs] for global_global,
                        tensor of shape [n_nodes, n_graphs] for local_global.
        positive (bool): Set True if the d_prime is masked for positive pairs,
                        set False for negative pairs.
    '''
    log_2 = np.log(2.)
    if positive:
        score = log_2 - F.softplus(-masked_d_prime)
    else:
        score = F.softplus(-masked_d_prime) + masked_d_prime - log_2
    return score



def NCE_loss(zs=None, zs_n=None, batch=None, sigma=None, **kwargs):
    '''The InfoNCE (NT-XENT) loss in contrastive learning.
    
    Args:
        zs (list, optipnal): List of tensors of shape [batch_size, z_dim].
        zs_n (list, optional): List of tensors of shape [nodes, z_dim].
        batch (Tensor, optional): Required when both :obj:`zs` and :obj:`zs_n` are given.
        sigma (ndarray, optional): A 2D-array of shape [:obj:`n_views`, :obj:`n_views`] with boolean 
            values, indicating contrast between which two views are computed. Only required 
            when number of views is greater than 2. If :obj:`sigma[i][j]` = :obj:`True`, 
            infoNCE between :math:`view_i` and :math:`view_j` will be computed.
        tau (int, optional): The temperature used in NT-XENT.

    :rtype: :class:`Tensor`
    '''
    assert zs is not None or zs_n is not None
    
    if 'tau' in kwargs:
        tau = kwargs['tau']
    else:
        tau = 0.5
    
    if 'norm' in kwargs:
        norm = kwargs['norm']
    else:
        norm = True
    
    mean = kwargs['mean'] if 'mean' in kwargs else True
        
    if zs_n is not None:
        if zs is None:
            # InfoNCE in GRACE
            assert len(zs_n)==2
            return (infoNCE_local_intra_node(zs_n[0], zs_n[1], tau, norm, batch)+
                    infoNCE_local_intra_node(zs_n[1], zs_n[0], tau, norm, batch))*0.5
        else:
            assert len(zs_n)==len(zs)
            assert batch is not None
            
            if len(zs)==1:
                return infoNCE_local_global(zs[0], zs_n[0], batch, tau, norm)
            elif len(zs)==2:
                return (infoNCE_local_global(zs[0], zs_n[1], batch, tau, norm)+
                        infoNCE_local_global(zs[1], zs_n[0], batch, tau, norm))
            else:
                assert len(zs)==len(sigma)
                loss = 0
                for (i, j) in itertools.combinations(range(len(zs)), 2):
                    if sigma[i][j]:
                        loss += (infoNCE_local_global(zs[i], zs_n[j], batch, tau, norm)+
                                 infoNCE_local_global(zs[j], zs_n[i], batch, tau, norm))
                return loss
    
    if len(zs)==2:
        return NT_Xent(zs[0], zs[1], tau, norm)
    elif len(zs)>2:
        assert len(zs)==len(sigma)
        loss = 0
        for (i, j) in itertools.combinations(range(len(zs)), 2):
            if sigma[i][j]:
                loss += NT_Xent(zs[i], zs[j], tau, norm)
        return loss


    
def infoNCE_local_intra_node(z1_n, z2_n, tau=0.5, norm=True, batch=None):
    '''
    Args:
        z1_n: Tensor of shape [n_nodes, z_dim].
        z2_n: Tensor of shape [n_nodes, z_dim].
        tau: Float. Usually in (0,1].
        norm: Boolean. Whether to apply normlization.
        batch: Tensor of shape [batch_size]
    '''
    def sim(z1:torch.Tensor, z2:torch.Tensor):
            if norm:
                z1 = F.normalize(z1)
                z2 = F.normalize(z2)
            return torch.mm(z1, z2.t())
    
    exp = lambda x: torch.exp(x / tau)
    if batch is not None:
        batch_size = batch.size(0)
        num_nodes = z1_n.size(0)
        indices = torch.arange(0, num_nodes).to(z1_n.device)
        losses = []
        for i in range(0, num_nodes, batch_size):
            mask = indices[i:i+batch_size]
            refl_sim = exp(sim(z1_n[mask], z1_n))
            between_sim = exp(sim(z1_n[mask], z2_n))
            losses.append(-torch.log(between_sim[:, i:i+batch_size].diag()
                            / (refl_sim.sum(1) + between_sim.sum(1)
                            - refl_sim[:, i:i+batch_size].diag())))
        losses = torch.cat(losses)
        return losses.mean()

    refl_sim = exp(sim(z1_n, z1_n))
    between_sim = exp(sim(z1_n, z2_n))
    
    pos_sim = between_sim.diag()
    intra_sim = refl_sim.sum(1) - refl_sim.diag()
    inter_pos_sim = between_sim.sum(1)
    
    loss = pos_sim / (intra_sim + inter_pos_sim)
    loss = -torch.log(loss).mean()

    return loss
    
    
                
def infoNCE_local_global(z_n, z_g, batch, tau=0.5, norm=True):
    '''
    Args:
        z_n: Tensor of shape [n_nodes, z_dim].
        z_g: Tensor of shape [n_graphs, z_dim].
        tau: Float. Usually in (0,1].
        norm: Boolean. Whether to apply normlization.
    '''
    # Not yet used in existing methods, to be implemented.
    loss = 0

    return loss



def NT_Xent(z1, z2, tau=0.5, norm=True):
    '''
    Args:
        z1, z2: Tensor of shape [batch_size, z_dim]
        tau: Float. Usually in (0,1].
        norm: Boolean. Whether to apply normlization.
    '''
    
    batch_size, _ = z1.size()
    sim_matrix = torch.einsum('ik,jk->ij', z1, z2)
    
    if norm:
        z1_abs = z1.norm(dim=1)
        z2_abs = z2.norm(dim=1)
        sim_matrix = sim_matrix / torch.einsum('i,j->ij', z1_abs, z2_abs)
        
    sim_matrix = torch.exp(sim_matrix / tau)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()
    return loss