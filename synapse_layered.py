from brian2 import *
from util import *
from visualize import *
import random
from constant import *
from layers import *

def run_and_update(net, synapses, time):
    net.run(time)
    # for S in synapses:
    #     S.connect(p=agg_connect_probability/100)

def simulate_layers(image, numberOfLayers, label, image_counter):

    if isinstance(image, list):
        (inputGroup ,input_spikeMonitor, networkOperation) = poisson_encoding_images(image, max_rate)
    else:
        (inputGroup ,input_spikeMonitor, networkOperation) = poisson_encoding(image, max_rate)

    net = Network(inputGroup, input_spikeMonitor)
    spikeMonitors = [input_spikeMonitor]
    neuronGroups = [inputGroup]
    synapses = []

    meta_collection = (net, spikeMonitors, neuronGroups, synapses)

    sobel_y_kernel = [1,2,1,0,0,0,-1,-2,-1]
    sobel_x_kernel = [1,0,-1,2,0,-2,1,0,-1]

    (meta_collection, convGroup1) = generate_connect_layers_conv(inputGroup, meta_collection, stride=1, kernel_data=sobel_y_kernel)
    (meta_collection, convGroup2) = generate_connect_layers_conv(inputGroup, meta_collection, stride=1, kernel_data=sobel_x_kernel)

    convGroups = [convGroup1, convGroup2]
    # convGroups = [convGroup1]

    # generate 1 less layers, cuz input layer count as 1
    meta_collection = generate_connect_layers(convGroups, numberOfLayers-2, meta_collection)
    (net, spikeMonitors, neuronGroups, synapses) = meta_collection

    net.add(networkOperation)

    net.run(total_duration)

    # collect the number of neurons in each layer
    img_neurons = [neuron_group.N for neuron_group in neuronGroups]

    visualize_multi_layer_spikes_2D(spikeMonitors, img_neurons, total_duration_graph, interval, label, len(spikeMonitors), image_counter)

    weight_matrices = visualize_multi_layer_weights_2D(synapses, label, len(spikeMonitors), image_counter)

    print(f"[{image_counter}] image is finished. {numberOfLayers} layers.")
    return weight_matrices



