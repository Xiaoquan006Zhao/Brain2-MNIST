from brian2 import *
from util import *
from visualize import *
import random
from constant import *
from layers import *


def generate_layers(G1 ,numberOfLayersNeed, net, synapses, spikeMonitors):
    N = G1.N
    layers = [G1]

    for count in range(numberOfLayersNeed):
        # numberOfNeurons = rand()*1.5*N
        # if numberOfNeurons < 0.5*N:
        #     numberOfNeurons = N
        layers.append(NeuronGroup(N, neuron_eqs, threshold=threshold_eqs, reset=reset_eqs, refractory=tau/20, method='exact'))
        new_layer = layers[-1]
        new_layer.v = 0
        new_layer.theta = 0

        spike_monitor = SpikeMonitor(new_layer)
        spikeMonitors.append(spike_monitor)
        net.add(spike_monitor)
        net.add(new_layer)

        (net, synapses) = connect_layers_excitory(layers[count], layers[count+1], input_connect_probability, net, synapses)

    return (net, synapses, spikeMonitors)

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
    synapses = []
    spikeMonitors = [input_spikeMonitor]

    (net, spikeMonitors, convGroup) = generate_connect_layers_conv(inputGroup, net, spikeMonitors)

    # generate 1 less layers, cuz input layer count as 1
    (net, synapses, spikeMonitors) = generate_layers(convGroup, numberOfLayers-2, net, synapses, spikeMonitors)
    net.add(networkOperation)

    net.run(total_duration)

    # for iter in range(iteration):
    #     run_and_update(net, synapses, duration)

    # collect the number of neurons in each layer
    img_neurons = [int(S.source.N) for S in synapses]
    img_neurons.append(int(synapses[-1].target.N))

    visualize_multi_layer_spikes_2D(spikeMonitors, img_neurons, total_duration_graph, interval, label, len(spikeMonitors), image_counter)

    weight_matrices = visualize_multi_layer_weights_2D(synapses, label, len(spikeMonitors), image_counter)

    print(f"[{image_counter}] image is finished. {numberOfLayers} layers.")
    return weight_matrices



