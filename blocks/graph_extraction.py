import tensorflow as tf
import networkx as nx
import numpy as np

def keras_to_digraph(model_path, output_path='network.gexf'):
    """
    Extract a directed graph from a Keras model where:
    - Nodes represent neurons
    - Edges represent connections with weights

    Args:
        model_path: Path to the .keras file
        output_path: Path for output GEXF file
    """
    model = tf.keras.models.load_model(model_path)

    G = nx.DiGraph()

    node_id = 0
    layer_node_map = {}  # (layer_idx, neuron_idx) mapping

    for layer_idx, layer in enumerate(model.layers):
        try:
            if hasattr(layer, 'output_shape'):
                output_shape = layer.output_shape
            elif hasattr(layer, 'output'):
                output_shape = layer.output.shape
            else:
                continue
        except:
            continue

        if isinstance(output_shape, tuple):
            if len(output_shape) > 1:
                num_neurons = int(np.prod(output_shape[1:]))
            else:
                num_neurons = 1
        else:
            num_neurons = 1

        layer_node_map[layer_idx] = {}

        for neuron_idx in range(num_neurons):
            G.add_node(
                node_id,
                layer=layer_idx,
                layer_name=layer.name,
                neuron_index=neuron_idx,
                label=f"L{layer_idx}_N{neuron_idx}"
            )
            layer_node_map[layer_idx][neuron_idx] = node_id
            node_id += 1

    prev_layer_size = None
    prev_layer_idx = None

    for layer_idx, layer in enumerate(model.layers):
        weights = layer.get_weights()

        if len(weights) > 0:
            # weights[0] is typically the kernel (weight matrix)
            kernel = weights[0]

            if len(kernel.shape) == 2:
                # dense layer
                input_size, output_size = kernel.shape

                # edges from previous layer to current layer
                if prev_layer_idx is not None:
                    for i in range(input_size):
                        for j in range(output_size):
                            weight_value = float(kernel[i, j])

                            if i in layer_node_map[prev_layer_idx] and \
                               j in layer_node_map[layer_idx]:
                                source_node = layer_node_map[prev_layer_idx][i]
                                target_node = layer_node_map[layer_idx][j]

                                G.add_edge(
                                    source_node,
                                    target_node,
                                    weight=weight_value
                                )

            elif len(kernel.shape) == 4:
                # convolutional layer (height, width, in_channels, out_channels)
                h, w, in_ch, out_ch = kernel.shape

                # for conv layers, create aggregate connections
                # between input and output feature maps
                if prev_layer_idx is not None:
                    for i in range(min(len(layer_node_map[prev_layer_idx]), in_ch)):
                        for j in range(min(len(layer_node_map[layer_idx]), out_ch)):
                            weight_value = float(np.mean(kernel[:, :, i, j]))

                            if i in layer_node_map[prev_layer_idx] and \
                               j in layer_node_map[layer_idx]:
                                source_node = layer_node_map[prev_layer_idx][i]
                                target_node = layer_node_map[layer_idx][j]

                                G.add_edge(
                                    source_node,
                                    target_node,
                                    weight=weight_value
                                )

        if layer_idx in layer_node_map and len(layer_node_map[layer_idx]) > 0:
            prev_layer_idx = layer_idx
            prev_layer_size = len(layer_node_map[layer_idx])

    nx.write_gexf(G, output_path)

    print(f"output path: {output_path}")
    print(f"num of nodes: {G.number_of_nodes()}")
    print(f"num of edges: {G.number_of_edges()}")
    print(f"num of layers: {len(model.layers)}")

    return G