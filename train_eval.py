import time
import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, nfeat, labels, val_nid, device, batch_size, num_workers):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, nfeat, device, batch_size, num_workers)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid].to(pred.device))

def evaluate_test(model, g, nfeat, labels, val_nid, test_nid, device, batch_size, num_workers):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    test_nid : the node Ids for testing.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, nfeat, device, batch_size, num_workers)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid].to(pred.device)), \
        compute_acc(pred[test_nid], labels[test_nid].to(pred.device))

def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a subset of nodes
    """
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels

def train_sample(model, data, hyperparams, device, eval_every):
    train_g, val_g, test_g, train_nfeat, train_labels, val_nfeat, val_labels, test_nfeat, test_labels = data
    train_nid = th.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = th.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = th.nonzero(~(test_g.ndata['train_mask'] | test_g.ndata['val_mask']), as_tuple=True)[0]

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(hyperparams['fanouts'])
    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        batch_size=hyperparams['batch_size'],
        shuffle=True,
        drop_last=False,
        num_workers=hyperparams['num_workers'])

    # Define model and optimizer
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['lr'])

    # Training loop
    avg = 0
    iter_tput = []
    best_val = best_test = 0.0
    for epoch in range(hyperparams['num_epochs']):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        tic_step = time.time()
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor(train_nfeat, train_labels,
                                                        seeds, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % 20 == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                    epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))
            tic_step = time.time()

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 1:
            avg += toc - tic
        if epoch % eval_every == 0 and epoch != 0:
            val_acc, test_acc = evaluate_test(model, test_g, test_nfeat, test_labels, val_nid, test_nid, device,
                                hyperparams['eval_batch_size'], hyperparams['num_workers'])
            print('Val acc {:.4f}, Test Acc: {:.4f}'.format(val_acc, test_acc))
            if val_acc >= best_val:
                best_val = val_acc
                best_test = test_acc
                print('Best val acc {:.4f}, best test acc: {:.4f}'.format(best_val, best_test))

    print('Avg epoch time: {}'.format(avg / (epoch - 4)))
    print('Best val acc {:.4f}, best test acc: {:.4f}'.format(best_val, best_test))

def load_subtensor1(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a subset of nodes
    """
    batch_inputs = nfeat(input_nodes).to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels
    
def train_ft_embed(model, data, hyperparams, device, eval_every):
    g, nfeat, labels = data
    train_nid = th.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = th.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = th.nonzero(~(g.ndata['train_mask'] | g.ndata['val_mask']), as_tuple=True)[0]

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(hyperparams['fanouts'])
    dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nid,
        sampler,
        batch_size=hyperparams['batch_size'],
        shuffle=True,
        drop_last=False,
        num_workers=hyperparams['num_workers'])

    # Define model and optimizer
    def initializer(emb):
        emb[:] = nfeat
        return emb
    embed = dgl.nn.NodeEmbedding(g.number_of_nodes(), nfeat.shape[1], name='input_embed',
                                 init_func=initializer, device=device)
    print('embedding is in ', embed.emb_tensor.device)
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['lr'])
    emb_optimizer = dgl.optim.SparseAdam(params=[embed], lr=hyperparams['sparse_lr'], eps=1e-8)

    # Training loop
    avg = 0
    iter_tput = []
    best_val = best_test = 0.0
    for epoch in range(hyperparams['num_epochs']):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        tic_step = time.time()
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor1(embed, labels,
                                                         seeds, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            emb_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            emb_optimizer.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % 20 == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                    epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))
            tic_step = time.time()

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 1:
            avg += toc - tic
        if epoch % eval_every == 0 and epoch != 0:
            val_acc, test_acc = evaluate_test(model, g, embed.emb_tensor, labels, val_nid, test_nid, device,
                                hyperparams['eval_batch_size'], hyperparams['num_workers'])
            print('Val acc {:.4f}, Test Acc: {:.4f}'.format(val_acc, test_acc))
            if val_acc >= best_val:
                best_val = val_acc
                best_test = test_acc
                print('Best val acc {:.4f}, best test acc: {:.4f}'.format(best_val, best_test))

    print('Avg epoch time: {}'.format(avg / (epoch - 4)))
    print('Best val acc {:.4f}, best test acc: {:.4f}'.format(best_val, best_test))