from brian2 import *
from util import *
from visualize import *
import random
from synapse_layered import run_and_update, connect_layers_excitory
from constant import *


def generate_agg(G1, G1_spikeMonitor ,multiplierOfN):
    N = G1.N

    net = Network(G1, G1_spikeMonitor)
    synapses = []
    spikeMonitors = [G1_spikeMonitor]
    layers = [G1]

    agg_group = NeuronGroup(multiplierOfN*N, eqs, threshold='v>1', reset='v=0', refractory=1*tau, method='exact')
    agg_group.v = 0

    agg_spikeMonitor = SpikeMonitor(agg_group)

    layers.append(agg_group)
    spikeMonitors.append(agg_spikeMonitor)
    net.add(agg_spikeMonitor)
    net.add(agg_group)

    G_prev = layers[0]
    G_next = layers[1]

    (net, synapses) = connect_layers_excitory(G_prev, G_next, input_connect_probability, net, synapses)
    (net, synapses) = connect_layers_excitory(G_next, G_next, agg_connect_probability, net, synapses)
    # (net, synapses) = connect_layers_inhibitory(G_next, G_next, agg_connect_probability/10, net, synapses)
    
    return (net, synapses, spikeMonitors)

def simulate_agg(image, multiplierOfN, label, image_counter):
    (inputGroup ,input_spikeMonitor, networkOperation) = poisson_encoding(image, max_rate)

    (net, synapses, spikeMonitors) = generate_agg(inputGroup, input_spikeMonitor, multiplierOfN)

    net.add(networkOperation)

    for iter in range(iteration):
        run_and_update(net, synapses, duration)

    visualize_multi_layer_spikes_2D(spikeMonitors, [(29, 28), (29*multiplierOfN, 28)], total_duration_graph, interval, label, len(spikeMonitors), image_counter)
    weight_matrices = visualize_multi_layer_weights_2D(synapses, label, len(spikeMonitors), image_counter)

    print(f"[{image_counter}] image is finished. {multiplierOfN} * N")
    return weight_matrices



