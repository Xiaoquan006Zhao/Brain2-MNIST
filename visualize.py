import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from util import all_alphabets, perfect_square
from brian2 import ms

save_path = f'/Users/zhaoxiaoquan/Documents/Brain2 MNIST/gif/'

def visualize_multi_layer_spikes_2D(spike_monitors,  img_neurons, duration, interval, label, max_layer, image_counter):
    img_shapes = [perfect_square(number_neuron) for number_neuron in img_neurons]
    
    fig = plt.figure(figsize=(8, 6))  
    gs = GridSpec(1, len(img_neurons), width_ratios=[shape[1] for shape in img_shapes])

    axs = [fig.add_subplot(gs[i], aspect='equal') for i in range(len(img_neurons))]

    # fig, axs = plt.subplots(1, len(spike_monitors))  
    ims = [ax.imshow(np.zeros(img_shape), cmap='gray', vmin=0, vmax=1) for ax, img_shape in zip(axs, img_shapes)]
    last_time = 0

    # Adding headers A-Z over the first 26 columns
    for i, number_tag in enumerate(all_alphabets):
        axs[0].text(i+1, -1, number_tag, ha='center', va='center')

    def update(frame):
        # re-initialize every time for fresh update, because when the previous effect of activation stack up
        # it's hard to visualize what is happening
        spike_counters = [np.zeros(np.prod(img_shape)) for img_shape in img_shapes]
        max_spikes = 0
        nonlocal last_time

        for spike_index in range(len(spike_monitors)):

            spike_monitor = spike_monitors[spike_index]
            spike_times = spike_monitor.t/ms
            spike_indices = spike_monitor.i
            current_time = frame * interval

            active_pixels = spike_indices[(last_time < spike_times) & (spike_times <= current_time)]
            # only update time meta variable if all spike monitors have been recorded
            if spike_index == len(spike_monitors) -1:
                last_time = current_time

            zero_activation = True
            
            for p in active_pixels:
                zero_activation = False
                spike_counters[spike_index][p] += 1

            max_spikes = [np.max(spike_counter) for spike_counter in spike_counters]
            max_spikes = np.max(max_spikes)

            local_max_spikes = np.max(spike_counters[spike_index])

            # set title for sub-plots
            axs[spike_index].set_title(f'')
            if(local_max_spikes < 0.3 * max_spikes):
                if local_max_spikes == 0 and zero_activation == True:
                    axs[spike_index].set_title(f'i:{spike_index}? active: {len(active_pixels)}')
                else:
                    axs[spike_index].set_title(f'coloring')
                max_spikes = local_max_spikes
            
            # To avoid division by zero
            if max_spikes > 0:  
                img = spike_counters[spike_index] / max_spikes
            else:
                img = np.zeros(np.prod(img_shapes[spike_index]))

            img = img.reshape(img_shapes[spike_index])
            ims[spike_index].set_array(img)
        
        max_spikes = [np.max(spike_counter) for spike_counter in spike_counters]
        max_spikes = np.max(max_spikes)
        fig.suptitle(f"Frame: {frame}, Max spikes: {max_spikes}") # +1 because count is 0-based
        return ims

    # frames=range(int(duration // interval) + 1), +1 because range is exclusive of the last index
    ani = animation.FuncAnimation(fig, update, frames=range(int(duration // interval) + 1), blit=True)
    
    ani.save(f'{save_path}label-{label}-max-{max_layer}-vari-{image_counter}.gif', writer='pillow')
    # plt.show()

def visualize_multi_layer_weights_2D(synapses, label, max_layer, image_counter):
    fig, axs = plt.subplots(1, len(synapses)) 

    if len(synapses) == 1:
        axs = [axs]

    weight_matrices = []

    for index, synapse in enumerate(synapses):
        all_shape = (len(synapse.source), len(synapse.target))
        weight_matrix = np.zeros(all_shape)

        for i, j, weight in zip(synapse.i, synapse.j, synapse.w):
            weight_matrix[i][j] = weight
        
        print(f"index: {index}, Max weight {np.max(weight_matrix)}")

        weight_matrices.append(weight_matrix)
        
        im = axs[index].imshow(weight_matrix[:,:], cmap='inferno', vmin=0, vmax=1)
        axs[index].set_title(f'Layer {index+1} Weights')
    
    plt.savefig(f'{save_path}Weight-label-{label}-max-{max_layer}-vari-{image_counter}.png', dpi=1000)
    # plt.show()

    return weight_matrices