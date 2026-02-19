# Prefect Flow Architecture
## Training of NN and Extraction of Graphs
### List items are individual Prefect tasks

1. fetch_data_and_configs()
   ├── Load dataset YAML
   ├── Load model config YAMLs (normal, binary, ternary)
   └── Split data into 10%, 50%, 100% while keeping the class ratio

2. for dataset_size in [10%, 50%, 100%]:
   ├── train_normal_nn(dataset_size)
   └── save model file (Keras, ONNX)

3. train_binary_nn()
   ├── Replacing Dense with BinaryDense
   └── save model file (Keras, ONNX)

4. train_ternary_nn()
   ├── Replacing Dense with TernaryDense
   └── save model file (Keras, ONNX)

5. prune_normal_nn()
   ├── Prune 100% normal network
   └── save model file (Keras, ONNX)

6. for model in [normal, binary, ternary, pruned]:
   ├── extract_nx_digraph(model)
   └── save to gexf and CytoScape comaptible file

7. for graph in [normal, binary, ternary, pruned]:
   ├── Apply Graph2Vec for later comparison
   └── Save Vector to `.npy`
