from model_constant_simple import *
from brian2 import *
from visualize import visualize_multi_layer_spikes_2D
from constant import *

def test_generation(trained_synapse, img_neuron):
    test_group = NeuronGroup(img_neuron, neuron_eqs, threshold="v > 1", reset=reset_eqs, method='exact')
    test_group_monitor = SpikeMonitor(test_group)

    S = Synapses(test_group, test_group, 
                 model=synapse_voltage_model+synapse_learning_model, 
                 on_pre= "v_post += w",
                 on_post= ""
                )
    
    S.connect(condition="i!=j" ,p=1)
    S.delay = 0.01*tau

    S.w = trained_synapse.w

    test_net = Network(S, test_group, test_group_monitor)

    test_net.run(1*ms)
    test_net.run(1*ms)

    for i in range(10):
        test_group.v = 0
        
        # @network_operation(dt=1*ms)
        # def input_test():
        #     nonlocal i
        #     test_group.v[i] = 300   
        # test_net.add(input_test)

        test_group.v[i] = 300   
        test_net.run(3*ms)
        test_group.v = 0
        test_net.run(1*ms)

    # print(test_group_monitor.i)

    visualize_multi_layer_spikes_2D([test_group_monitor], [img_neuron], 42, 1)
