# Graph-theoretic analysis of Neural Networks

## Stack
![Prefect](https://img.shields.io/badge/-Prefect-darkgreen?logo=Prefect&style=for-the-badge&color=070E10) ![YAML](https://img.shields.io/badge/-yaml-CB171E?logo=yaml&style=for-the-badge)

![NetworkX](https://img.shields.io/badge/-NetworkX-darkgreen?logo=NetworkX&style=for-the-badge)
![Nvidia](https://img.shields.io/badge/nx--cugraph-000000.svg?style=for-the-badge&logo=nVIDIA&logoColor=green)
![Keras](https://img.shields.io/badge/-Keras-D00000?logo=keras&style=for-the-badge)
![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&style=for-the-badge)
![PyG](https://img.shields.io/badge/-PyG-3C2179?logo=numpy&style=for-the-badge)

![CoLab](https://img.shields.io/badge/-Google%20Colab-013243?logo=googlecolab&style=for-the-badge)
![VSCode](https://img.shields.io/badge/-Visual%20Studio%20Code-1062FB?logo=codecrafters&style=for-the-badge)

## Structure
- ***blocks*** - building blocks, utility codes such as: pruner, quantized layers
- ***configs*** - YAML configuration files, such as: model configs, pruning parameters
- ***flows*** - Prefect flows handling the *data -> model -> graph* pipeline
- ***plans*** - The architectural plans of the three parts of the analysis (network extraction, network visualization and the visualization of the findings)