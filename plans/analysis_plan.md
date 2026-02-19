# Prefect Flow Architecture
## Graph Comparison and network Visualization
### List items are individual Prefect task

1. Compare embeddings for a given dataset
   ├── Load the 10-50% networks, compare to 100%
   ├── Load the Binary and Ternary networks, compare to Original
   ├── Load the Pruned network, compare to Original
   └── Return the distances in a dict

2. Visualize networks
   ├── nx-cugraph -> force_atlas2
   └── Export