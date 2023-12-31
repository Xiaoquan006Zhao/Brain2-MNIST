from brian2 import *
from util import *
from visualize import *
import random
from constant import *

def connect_layers_excitory(G1, G2, connection_probability, net, synapses):
    S = Synapses(G1, G2,
            '''
            w : 1
            dCa/dt = -Ca/tau_Ca : 1 (event-driven)

            dapre/dt = -apre/taupre : 1 (event-driven)
            dapost/dt = -apost/taupost : 1 (event-driven)
            ''',

            # v_post += w + condition_high_voltage*v_post*0.1, make it easier to fire when close to fire
                # like opening more voltage gates when close to fire
            # (100 * (Ca-0.2) * (0.4-Ca)) meanings center at 0.3, just to scale the LTD 
            # (Ca - 0.7) * 3, same reason as above
            on_pre='''
            condition_high_voltage = int(v_post > 0.75)
            v_post += w + condition_high_voltage*v_post*0.1

            Ca = clip(Ca + w, CaMin, CaMax)
            
            condition_Remove_Mg = int(v_post > 0.7)
            condition_LTD = int(Ca > 0.2 and Ca < 0.4)
            condition_LTP = int(Ca > 0.7)

            w -= wIncrement * condition_Remove_Mg * condition_LTD * (100 * (Ca-0.2) * (0.4-Ca))
            w += wIncrement * condition_Remove_Mg * condition_LTP * (Ca - 0.7) * 3
            
            apre += Apre
            w -= apost

            w = clip(w, wMin, wMax)
            ''',
            on_post='''
            condition_high_voltage = int(v_pre > 0.75)
            v_pre += w + condition_high_voltage*v_post*0.1

            apost += Apost
            w += apre
            
            w = clip(w, wMin, wMax)

            '''
    )

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
    S.w = 'wStart*rand()'
        
    return (net, synapses)

def generate_layers(G1, G1_spikeMOnitor ,numberOfLayersNeed):
    N = G1.N

    net = Network(G1, G1_spikeMOnitor)
    synapses = []
    spikeMonitors = [G1_spikeMOnitor]
    layers = [G1]

    for count in range(numberOfLayersNeed):
        # numberOfNeurons = rand()*1.5*N
        # if numberOfNeurons < 0.5*N:
        #     numberOfNeurons = N
        layers.append(NeuronGroup(N, eqs, threshold='v>1', reset='v=0', refractory=tau/20, method='exact'))
        new_layer = layers[-1]
        new_layer.v = 0

        spike_monitor = SpikeMonitor(new_layer)
        spikeMonitors.append(spike_monitor)
        net.add(spike_monitor)
        net.add(new_layer)

        # @network_operation(dt=0.2*tau)
        # def print_v():
        #     if np.max(new_layer.v) > 0:
        #         print(f"max v: {np.max(new_layer.v)}")

        # net.add(print_v)

        (net, synapses) = connect_layers_excitory(layers[count], layers[count+1], input_connect_probability, net, synapses)

    # for count in range(numberOfLayersNeed):
    #     G_prev = layers[count]
    #     G_next = layers[count+1]

    #     (net, synapses) = connect_layers_excitory(G_prev, G_next, 0.05, net, synapses)   

    return (net, synapses, spikeMonitors)

def run_and_update(net, synapses, time):
    net.run(time)
    # for S in synapses:
    #     S.connect(p=agg_connect_probability/100)

    # for S in synapses:
    #     synapse_detail = list(zip(S.i, S.j, S.w))
    #     random.shuffle(synapse_detail)
    #     add_offset = 0
    #     for i, j, weight in synapse_detail:
    #         if weight > (wAddThreshold-add_offset):
    #             all_possible_targets = set(range(len(S.target)))
    #             connected_targets = set(S.j[S.i == i])
    #             unconnected_targets = all_possible_targets - connected_targets
    #             if unconnected_targets:
    #                 add_offset = 0
    #                 new_target_index = int(rand()*len(unconnected_targets))
    #                 new_target = list(unconnected_targets)[new_target_index]
    #                 S.connect(i=i, j=new_target)
    #                 S.w[i, new_target] = 'wStart*rand()*0.5'
    #                 # break
    #         add_offset += 0.01

def simulate_layers(image, numberOfLayers, label, image_counter):

    if isinstance(image, list):
        (inputGroup ,input_spikeMonitor, networkOperation) = poisson_encoding_images(image, max_rate)
    else:
        (inputGroup ,input_spikeMonitor, networkOperation) = poisson_encoding(image, max_rate)

    # generate 1 less layers, cuz input layer count as 1
    (net, synapses, spikeMonitors) = generate_layers(inputGroup, input_spikeMonitor, numberOfLayers-1)
    net.add(networkOperation)

    net.run(total_duration)

    # for iter in range(iteration):
    #     run_and_update(net, synapses, duration)

    visualize_multi_layer_spikes_2D(spikeMonitors, [(29, 28)]*numberOfLayers, total_duration_graph, interval, label, len(spikeMonitors), image_counter)

    weight_matrices = visualize_multi_layer_weights_2D(synapses, label, len(spikeMonitors), image_counter)

    print(f"[{image_counter}] image is finished. {numberOfLayers} layers.")
    return weight_matrices



