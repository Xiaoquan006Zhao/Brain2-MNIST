from brian2 import *
from constant import *
from util import perfect_square

def connect_layers_excitory(G1, G2, connection_probability, net, synapses):
    S = Synapses(G1, G2, model=synapse_voltage_model+synapse_learning_model, 
                 on_pre=on_pre_model, 
                 on_post=on_post_model)

    # S = Synapses(G1, G2,
    #          '''
    #          w : 1
    #          dapre/dt = -apre/taupre : 1 (event-driven)
    #          dapost/dt = -apost/taupost : 1 (event-driven)
    #          ''',
    #          on_pre='''
    #          v_post += w
    #          apre += Apre
    #          w = clip(w+apost, 0, wMax)
    #          ''',
    #          on_post='''
    #          apost += Apost
    #          w = clip(w+apre, 0, wMax)
    #          ''')
        
    S.connect(p=connection_probability)
    net.add(S)
    synapses.append(S)

    for neuron_index in range(len(G1)):
        if not np.any(S.i[:] == neuron_index):
            target = randint(len(G2))
            S.connect(i=neuron_index, j=target)

    S.w = 'clip(wStart*rand(), wStart*0.5, wStart)'
    S.w = 'wStart'
        
    return (net, synapses)

def generate_connect_layers_conv(G, net, spikeMonitors, kernel_size=3, stride=1):
    (input_height, input_width) = perfect_square(G.N)

    output_height = int((input_height - kernel_size) / stride) + 1
    output_width = int((input_width - kernel_size) / stride) + 1     
    
    conv_neurons = NeuronGroup(output_height * output_width, neuron_eqs, threshold=threshold_eqs, reset=reset_eqs,method='exact')
    net.add(conv_neurons)
    spike_monitor = SpikeMonitor(conv_neurons)
    net.add(spike_monitor)
    spikeMonitors.append(spike_monitor)

    connect_layers_conv(G, net, conv_neurons, (kernel_size, output_height, output_width, input_height, input_width, stride))

    return (net, spikeMonitors, conv_neurons)

def connect_layers_conv(G, net, conv_neurons, meta_data):
    (kernel_size, output_height, output_width, input_height, input_width, stride) = meta_data
    kernel_synapses = []

    for kernel_row in range(kernel_size):
        for kernel_col in range(kernel_size):
            S = Synapses(G, conv_neurons,
                model='''
                w : 1 (shared)
                '''+synapse_learning_model,
                on_pre='''
                '''+on_pre_model,
                on_post='''
                '''+on_post_model)
            net.add(S)
            kernel_synapses.append(S)

    for output_index in range(conv_neurons.N):
        start_row = (output_index // output_width) * stride
        start_col = (output_index % output_width) * stride

        for kernel_row in range(kernel_size):
            for kernel_col in range(kernel_size):
                input_index = (start_row + kernel_row) * input_width + (start_col + kernel_col)
                kernel_synapses[kernel_row * kernel_size + kernel_col].connect(i=input_index, j=output_index)

    for ks in kernel_synapses:
        ks.w = rand()
            



