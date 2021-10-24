import torch as th
import torch.nn as nn
import torch.functional as F
import dgl
import dgl.nn as dglnn
import sklearn.linear_model as lm
import sklearn.metrics as skm
import tqdm
import torch.cuda.nvtx as nvtx

import os
from scipy.sparse.linalg import eigs
import numpy as np

class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.init(in_feats, n_hidden, n_classes, n_layers, activation, dropout)

    def init(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if n_layers > 1:
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        else:
            self.layers.append(dglnn.SAGEConv(in_feats, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, device, batch_size, num_workers):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.num_nodes()).to(g.device),
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
        return y

class GAT(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 num_classes,
                 num_layers,
                 num_heads,
                 activation,
                 dropout):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.n_hidden = num_hidden
        self.n_heads = num_heads
        self.n_classes = num_classes
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        # input projection (no residual)
        self.gat_layers.append(dglnn.GATConv(
            in_dim, num_hidden, num_heads, activation=self.activation, allow_zero_in_degree=True))
        # hidden layers
        for l in range(1, num_layers - 1):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(dglnn.GATConv(
                num_hidden * num_heads, num_hidden, num_heads, activation=self.activation, allow_zero_in_degree=True))
        # output projection
        self.gat_layers.append(dglnn.GATConv(
            num_hidden * num_heads, num_classes, num_heads, allow_zero_in_degree=True))

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.gat_layers, blocks)):
            nvtx.range_push('gat-layer-' + str(l))
            h = layer(block, h)
            nvtx.range_pop()
            if l != len(self.gat_layers) - 1:
                h = h.flatten(1)
                h = self.dropout(h)
        h = h.mean(1)
        return h

    def inference(self, g, x, device, batch_size, num_workers):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        for l, layer in enumerate(self.gat_layers):
            y = th.zeros(g.num_nodes(), self.n_hidden * self.n_heads if l != len(self.gat_layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.num_nodes()),
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.gat_layers) - 1:
                    h = h.flatten(1)
                else:
                    h = h.mean(1)
                y[output_nodes] = h.cpu()

            x = y
        return y


def compute_acc_unsupervised(emb, labels, train_nids, val_nids, test_nids):
    """
    Compute the accuracy of prediction given the labels.
    """
    emb = emb.cpu().numpy()
    labels = labels.cpu().numpy()
    train_nids = train_nids.cpu().numpy()
    train_labels = labels[train_nids]
    val_nids = val_nids.cpu().numpy()
    val_labels = labels[val_nids]
    test_nids = test_nids.cpu().numpy()
    test_labels = labels[test_nids]

    emb = (emb - emb.mean(0, keepdims=True)) / emb.std(0, keepdims=True)

    lr = lm.LogisticRegression(multi_class='multinomial', max_iter=10000)
    lr.fit(emb[train_nids], train_labels)

    pred = lr.predict(emb)
    f1_micro_eval = skm.f1_score(val_labels, pred[val_nids], average='micro')
    f1_micro_test = skm.f1_score(test_labels, pred[test_nids], average='micro')
    return f1_micro_eval, f1_micro_test

def get_eigen(g, k, name):
    if not os.path.exists('ogbn-products_eigenvals{}.npy'.format(k)):
        adj = g.adj(scipy_fmt='csr')
        start = time.time()
        eigen_vals, eigen_vecs = eigs(adj.astype(np.float32), k=k, tol=1e-5, ncv=k*3)
        print('Compute eigen: {:.3f} seconds'.format(time.time() - start))
        np.save('ogbn-products_eigenvals{}.npy'.format(k), eigen_vals)
        np.save('ogbn-products_eigenvecs{}.npy'.format(k), eigen_vecs)
    else:
        eigen_vals = np.load('ogbn-products_eigenvals{}.npy'.format(k))
        eigen_vecs = np.load('ogbn-products_eigenvecs{}.npy'.format(k))
        assert len(eigen_vals) == k
        assert eigen_vecs.shape[1] == k
    return eigen_vals, eigen_vecs