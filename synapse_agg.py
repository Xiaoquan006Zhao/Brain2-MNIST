from brian2 import *
from util import *
from visualize import *
import random
from synapse_layered import run_and_update, connect_layers_excitory
from constant import *
from layers import *
from test import *


def generate_agg(inputGroups, input_spikeMonitors ,multiplierOfN=1):
    input_group_excitory = inputGroups[0]
    input_spikeMonitor_excitory = input_spikeMonitors[0]

    input_group_inhibtory = inputGroups[1]
    input_spikeMonitor_inhibtory = input_spikeMonitors[1]

    N = input_group_excitory.N

    # Constructing Meta data collection
    net = Network(input_group_excitory, input_group_inhibtory, input_spikeMonitor_excitory, input_spikeMonitor_inhibtory)
    synapses = []
    spikeMonitors = [input_spikeMonitor_excitory, input_spikeMonitor_inhibtory]
    neuronGroups = [input_group_excitory, input_group_inhibtory]

    agg_group = NeuronGroup(multiplierOfN*N, neuron_eqs, threshold=threshold_eqs, reset=reset_eqs, refractory=0.2*tau, method='exact')
    agg_group.v = 0
    agg_group.theta = 0

    agg_spikeMonitor = SpikeMonitor(agg_group)
    neuronGroups.append(agg_group)
    spikeMonitors.append(agg_spikeMonitor)
    net.add(agg_group)
    net.add(agg_spikeMonitor)

    meta_collection = (net, spikeMonitors, neuronGroups, synapses)

    (net, synapses) = connect_layers_excitory(input_group_excitory, agg_group, 0, meta_collection)
    (net, synapses) = connect_layers_excitory(input_group_inhibtory, agg_group, 0, meta_collection, excitory_connection=False)

    (net, synapses) = connect_layers_excitory(agg_group, agg_group, 1, meta_collection)

    # @network_operation(dt=(iteration-15)*tau)
    # def reset_v():
    #     # agg_group.v[agg_group.v < -1000] = 0
    #     agg_group.v = 0

    # net.add(reset_v)
    
    return meta_collection

def simulate_agg(image, multiplierOfN, label, image_counter):
    # (inputGroups ,input_spikeMonitors, networkOperation) = poisson_encoding(image, max_rate)

    if isinstance(image, list):
        (inputGroups ,input_spikeMonitors, networkOperation) = poisson_encoding_images(image, max_rate)
    else:
        (inputGroups ,input_spikeMonitors, networkOperation) = poisson_encoding(image, max_rate)

    meta_collection = generate_agg(inputGroups, input_spikeMonitors, multiplierOfN)

    (net, spikeMonitors, neuronGroups, synapses) = meta_collection
    net.add(networkOperation)

    for iter in range(iteration):
        run_and_update(net, synapses, duration)

    img_neurons = [neuron_group.N for neuron_group in neuronGroups]

    visualize_multi_layer_spikes_2D(spikeMonitors, img_neurons, total_duration_graph, interval, label, len(spikeMonitors), image_counter)
    # weight_matrices = visualize_multi_layer_weights_2D(synapses, label, len(spikeMonitors), image_counter)

    main_layer_synapse = synapses[-1]
    test_generation(main_layer_synapse, img_neurons[-1])

    print(f"[{image_counter}] image is finished. {multiplierOfN} * N")
    # return weight_matrices



