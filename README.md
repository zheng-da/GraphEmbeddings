# GraphEmbeddings
This repo compares different techniques of using GNN models to train node embeddings of a graph without node/edge attributes.
Here we try a few options of adding initial node embeddings for GNN models:
* compute eigenvectors of the graph with SVD and use the eigenvectors as the positional node embeddings of the graph,
* put trainable embeddings of the nodes and initialize the embeddings with normal distribution,
* put trainable embeddings of the nodes and initialize the embeddings with eigenvectors.

We try these options with both GraphSage and GAT models.
