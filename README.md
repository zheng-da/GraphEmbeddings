# GraphEmbeddings
This repo compares different techniques of using GNN models to train node embeddings of a graph without node/edge attributes.
Here we try a few options of adding initial node embeddings for GNN models:
* Method 1: compute eigenvectors of the graph with SVD and use the eigenvectors as the positional node embeddings of the graph,
* Method 2: put trainable embeddings of the nodes and initialize the embeddings with normal distribution,
* Method 3: put trainable embeddings of the nodes and initialize the embeddings with eigenvectors.

We try these options with both GraphSage and GAT models. Below shows the model accuracy on OGBN-products graph.
| Method  | Val/Test Acc |
| ------------- | ------------- |
| MLP on eigenvectors  | 0.6229/0.3987 |
| GraphSage + original node features | 0.9201/0.7832 |
| GraphSage + trainable embeddings   | 0.9133/0.7207 |
| GraphSage + eigenvectors  | 0.8650/0.7015 |
| GraphSage + trainable embeddings init with eigenvectors | 0.9148/0.7898 |
| GraphSage + fine-tune embeddings init with eigenvectors | 0.9164/0.7900 |
| GAT + trainable embeddings | 0.9205/0.7622 |
| GAT + eigenvectors | 0.8657/0.7326 |
| GAT + trainable embeddings init with eigenvectors | 0.9177/0.8028 |
| GAT + fine-tune embeddings init with eigenvectors | 0.9158/0.8001 |
