from brian2 import *
from constant import *
from model_constant import *
from util import perfect_square

def add_to_meta_collection(neuronGroup, meta_collection):
    (net, spikeMonitors, neuronGroups, synapses) = meta_collection

    spike_monitor = SpikeMonitor(neuronGroup)
    spikeMonitors.append(spike_monitor)
    net.add(spike_monitor)
    net.add(neuronGroup)
    neuronGroups.append(neuronGroup)

    return (net, spikeMonitors, neuronGroups, synapses)

def initialize(neuronGroup):
    neuronGroup.v = 0
    neuronGroup.theta = 0

def generate_connect_layers(groups ,numberOfLayersNeed, meta_collection):
    (net, spikeMonitors, neuronGroups, synapses) = meta_collection

    if isinstance(groups, list):
        N = groups[0].N
        layers = []
        for group in groups:
            layers.append(group)

        layers.append(NeuronGroup(N, neuron_eqs, threshold=threshold_eqs, reset=reset_eqs, refractory=tau/20, method='exact'))
        new_layer = layers[-1]
        initialize(new_layer)
        meta_collection = add_to_meta_collection(new_layer, meta_collection)
        
        for group in groups:
            (net, synapses) = connect_layers_excitory(group, new_layer, 1, meta_collection)
    else:
        N = groups.N
        layers = [groups]

    for count in range(numberOfLayersNeed-1):
        # numberOfNeurons = rand()*1.5*N
        # if numberOfNeurons < 0.5*N:
        #     numberOfNeurons = N
        layers.append(NeuronGroup(N, neuron_eqs, threshold=threshold_eqs, reset=reset_eqs, refractory=tau/20, method='exact'))
        new_layer = layers[-1]
        initialize(new_layer)

        meta_collection = add_to_meta_collection(new_layer, meta_collection)

        (net, synapses) = connect_layers_excitory(layers[count], layers[count+1], input_connect_probability, meta_collection)

    return meta_collection

def connect_layers_excitory(G1, G2, connection_probability, meta_collection):
    (net, spikeMonitors, neuronGroups, synapses) = meta_collection

    S = Synapses(G1, G2, model=synapse_voltage_model+synapse_learning_model, 
                 on_pre=on_pre_model, 
                 on_post=on_post_model)
        
    S.connect(p=connection_probability)
    net.add(S)
    synapses.append(S)

    for neuron_index in range(len(G1)):
        if not np.any(S.i[:] == neuron_index):
            target = randint(len(G2))
            S.connect(i=neuron_index, j=target)

    S.w = 'clip(wStart*rand(), wStart*0.5, wStart)'
    S.w = 'wStart*rand()'
    S.w = 'wStart'

    # for neuron_index in range(len(G1)):
    #     synapse_indices = S.i[:] == neuron_index
    #     total_weight = np.sum(S.w[synapse_indices])
    #     if total_weight > 0:
    #         S.w[synapse_indices] /= total_weight / wMax

    return (net, synapses)

def generate_connect_layers_conv(G, meta_collection, kernel_size=3, stride=1, kernel_data=None):
    (net, spikeMonitors, neuronGroups, synapses) = meta_collection
    
    (input_height, input_width) = perfect_square(G.N)

    output_height = int((input_height - kernel_size) / stride) + 1
    output_width = int((input_width - kernel_size) / stride) + 1     
    
    conv_neurons = NeuronGroup(output_height * output_width, neuron_eqs, threshold=threshold_eqs, reset=reset_eqs,method='exact')
    net.add(conv_neurons)
    spike_monitor = SpikeMonitor(conv_neurons)
    net.add(spike_monitor)
    spikeMonitors.append(spike_monitor)
    neuronGroups.append(conv_neurons)

    meta_data = (kernel_size, output_height, output_width, input_height, input_width, stride)

    connect_layers_conv(G, net, conv_neurons, kernel_data, meta_data)
    
    return (meta_collection, conv_neurons)

def connect_layers_conv(G, net, conv_neurons, kernel_data, meta_data):
    (kernel_size, output_height, output_width, input_height, input_width, stride) = meta_data
    kernel_synapses = []

    for kernel_row in range(kernel_size):
        for kernel_col in range(kernel_size):
            S = Synapses(G, conv_neurons,
                model='''
                w : 1 (shared)
                '''+synapse_learning_model,
                on_pre='''
                v_post += w
                ''',
                on_post='''
                '''
                )
            net.add(S)
            kernel_synapses.append(S)

    for output_index in range(conv_neurons.N):
        start_row = (output_index // output_width) * stride
        start_col = (output_index % output_width) * stride

        for kernel_row in range(kernel_size):
            for kernel_col in range(kernel_size):
                input_index = (start_row + kernel_row) * input_width + (start_col + kernel_col)
                kernel_synapses[kernel_row * kernel_size + kernel_col].connect(i=input_index, j=output_index)

    # square kernel
    if kernel_data != None and len(kernel_data) == kernel_size**2:
        for index in range(len(kernel_data)):
                kernel_synapses[index].w = kernel_data[index]
            



