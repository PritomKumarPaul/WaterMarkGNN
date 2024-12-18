# Graph Neural Network Watermarking

## Overview
This project implements a method for watermarking Graph Neural Networks (GNNs) using a bi-level optimization technique. It applies a clean model trained on datasets like Cora or Pubmed and embeds watermarks into the model using a generator network. The approach is tested for its impact on the model's performance and its ability to verify the watermark effectively.

## Datasets
We utilize two commonly used datasets for graph learning:

- **Cora**: A citation network dataset with 2708 nodes, 5429 edges, and 7 classes.
- **Pubmed**: A larger dataset with 19717 nodes, 44338 edges, and 3 classes.

Both datasets are loaded using PyTorch Geometric's `Planetoid` class.

## Requirements
- Python
- PyTorch
- PyTorch Geometric
- Matplotlib
- Scikit-learn
- NetworkX


## Models
### 1. GCN (Graph Convolutional Network)
The primary GNN model architecture used for clean training and embedding watermarks.
```python
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
```

### 2. Generator
A model designed to generate the trigger graph and embed watermarks into the clean model.
```python
class Generator(nn.Module):
    def __init__(self, in_channels, hidden_channels, feature_dim):
        super(Generator, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, feature_dim)

    def forward(self, data, num_trigger_nodes):
        x, edge_index = data.x, data.edge_index
        trigger_features = F.relu(self.conv1(x, edge_index))
        trigger_features = self.conv2(trigger_features, edge_index)
        trigger_features = trigger_features[:num_trigger_nodes]

        adjacency_matrix = torch.randint(0, 2, (num_trigger_nodes, num_trigger_nodes)).float()
        adjacency_matrix = torch.triu(adjacency_matrix, diagonal=1).clone()
        adjacency_matrix = adjacency_matrix + adjacency_matrix.T.clone()

        return trigger_features, adjacency_matrix
```

## Key Steps

### Load Dataset
```python
from torch_geometric.datasets import Planetoid

# Load Cora dataset
dataset = Planetoid(root="/tmp/Cora", name="Cora")
data = dataset[0]

# Load Pubmed dataset
dataset_pubmed = Planetoid(root="/tmp/Pubmed", name="Pubmed")
data_pubmed = dataset_pubmed[0]
```

### Train GCN Model on Clean Data
```python
clean_model = GCN(dataset.num_node_features, 16, dataset.num_classes)
clean_loss_history = train_model(clean_model, data, epochs=200, lr=0.01)
```

### Bi-level Optimization for Watermark Embedding
```python
watermarked_model = GCN(dataset.num_node_features, 16, dataset.num_classes)
watermarked_model.load_state_dict(clean_model.state_dict())
watermarked_loss_history = bilevel_optimization(generator, watermarked_model, data, poisoned_nodes, epochs=200)
```

### Verify Watermark
```python
verify_watermark_accuracy(watermarked_model, data, poisoned_nodes)
```

### Compare Performance
The following code compares the performance of the clean and embedded models:
```python
# Evaluate clean and embedded models
y_true_clean, y_pred_clean = evaluate_model(clean_model, data, data.test_mask)
y_true_embed, y_pred_embed = evaluate_model(watermarked_model, data, data.test_mask)

# Visualization
def plot_performance_comparison(y_true_clean, y_pred_clean, y_true_embed, y_pred_embed):
    metrics_clean = {
        "Accuracy": accuracy_score(y_true_clean, y_pred_clean),
    }
    metrics_embed = {
        "Accuracy": accuracy_score(y_true_embed, y_pred_embed),
    }

    labels = list(metrics_clean.keys())
    clean_values = list(metrics_clean.values())
    embed_values = list(metrics_embed.values())

    x = range(len(labels))

    plt.bar(x, clean_values, width=0.4, label="Before Embedding", align="center")
    plt.bar([i + 0.4 for i in x], embed_values, width=0.4, label="After Embedding", align="center")
    plt.xticks([i + 0.2 for i in x], labels)
    plt.ylabel("Scores")
    plt.title("Performance Comparison")
    plt.legend()
    plt.show()

plot_performance_comparison(y_true_clean, y_pred_clean, y_true_embed, y_pred_embed)
```

## Results
- **Clean Model Performance**: The GCN model trained on the clean graph exhibits high accuracy and F1 scores.
- **Embedded Model Performance**: After watermark embedding, the model retains competitive accuracy while successfully embedding the watermark.
- **Watermark Verification**: The watermark verification process confirms the accuracy of the embedded watermark.

## Visualization
### Trigger Graph
The trigger graph generated during the embedding process can be visualized:
```python
visualize_trigger_graph(adjacency_matrix)
```

### Backdoor Graph
The complete graph with embedded backdoors can be visualized:
```python
visualize_backdoor_graph(data, poisoned_nodes, adjacency_matrix)
```
