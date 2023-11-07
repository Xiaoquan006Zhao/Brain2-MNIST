import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from util import all_alphabets
from brian2 import ms

save_path = f'/Users/zhaoxiaoquan/Documents/Brain2 MNIST/gif/'

def visualize_multi_layer_spikes_2D(spike_monitors,  img_shapes, duration, interval, label, max_layer, image_counter):
    fig, axs = plt.subplots(1, len(spike_monitors))  # Initialize subplots
    
    ims = [ax.imshow(np.zeros(img_shape), cmap='gray', vmin=0, vmax=1) for ax, img_shape in zip(axs, img_shapes)]

    spike_counters = [np.zeros(np.prod(img_shape)) for _, img_shape in zip(range(len(spike_monitors)), img_shapes)]
    
    last_time = 0

    # Adding headers A-Z over the first 26 columns
    for i, number_tag in enumerate(all_alphabets):
        axs[0].text(i+1, -1, number_tag, ha='center', va='center')

    def update(frame):
        max_spikes = 0
        nonlocal last_time
        for spike_index in range(len(spike_monitors)):
            spike_monitor = spike_monitors[spike_index]
            spike_times = spike_monitor.t/ms
            spike_indices = spike_monitor.i
            current_time = frame * interval
            active_pixels = spike_indices[(last_time < spike_times) & (spike_times <= current_time)]
            last_time = current_time

            for p in active_pixels:
                spike_counters[spike_index][p] += 1

            max_spikes = np.max(spike_counters)

            if max_spikes > 0:  # To avoid division by zero
                img = spike_counters[spike_index] / max_spikes
            else:
                img = np.zeros(np.prod(img_shapes[spike_index]))

            img = img.reshape(img_shapes[spike_index])
            ims[spike_index].set_array(img)
        
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
        
        print(f"wtf {np.max(weight_matrix)}")

        weight_matrices.append(weight_matrix)
        
        im = axs[index].imshow(weight_matrix[:,:], cmap='hot', vmin=0, vmax=1)
        axs[index].set_title(f'Layer {index+1} Weights')

    plt.savefig(f'{save_path}Weight-label-{label}-max-{max_layer}-vari-{image_counter}.png', dpi=4000)
    # plt.show()

    return weight_matrices