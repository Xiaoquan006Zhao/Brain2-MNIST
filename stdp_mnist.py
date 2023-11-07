from util import *
from brian2 import *
from itertools import combinations
from synapse_layered import simulate_layers
from synapse_agg import simulate_agg
from load_mnist import load_mnist

(X_separated, y_separated) = load_mnist()

def train_uniform_input_per_variation(number_label, variation, max_layers, multiplierOfN, cross_label, sample_indices):
    print("Start training")

    # sampling images from the MNIST dataset
    input_images = sample_images(X_separated, number_label, variation, sample_indices)

    if not multiplierOfN:
        weight_matrices_per_variation = [simulate_layers(image, max_layers, number_label, index) for index, image in enumerate(input_images)]
    else:
        weight_matrices_per_variation = [simulate_agg(image, multiplierOfN, number_label, index) for index, image in enumerate(input_images)]

    # transpose so that each subarray contains the corresponding synapse not just all synapse from one variation
    size_outer = len(weight_matrices_per_variation)
    size_inner = len(weight_matrices_per_variation[0])
    swapped_list = []
    for i in range(size_inner):
        swapped_list.append([weight_matrices_per_variation[j][i] for j in range(size_outer)])
    weight_matrices_per_variation = swapped_list

    if not cross_label:
        for index, layer in enumerate(weight_matrices_per_variation):
            pairwise_diffs = [np.abs(a - b) for a, b in combinations(layer, 2)]
            total_diff = [int(np.sum(diff)) for diff in pairwise_diffs]
            average = np.mean(total_diff)
            print(total_diff)
            print(average)
    else:
        return weight_matrices_per_variation

# at most two for simplicity of comprehension
def compare_train_uniform_input_per_variation(number_labels, variation, max_layers, multiplierOfN, sample_indices_two_label):    
    if not multiplierOfN:
        weight_matrices_per_label = [train_uniform_input_per_variation(label, variation, max_layers, multiplierOfN, True, sample_indices_label) for label, sample_indices_label in zip(number_labels, sample_indices_two_label)]
    else:
        weight_matrices_per_label = [train_uniform_input_per_variation(label, variation, max_layers, multiplierOfN, True, sample_indices_label) for label, sample_indices_label in zip(number_labels, sample_indices_two_label)]

    label1_weights, label2_weights = weight_matrices_per_label

    for index, label1_layer in enumerate(label1_weights):
        label2_layer = label2_weights[index]

        pairwise_diffs = [
            np.abs(w1 - w2) 
            for w1 in label1_layer 
            for w2 in label2_layer
        ]

        total_diff = [int(np.sum(diff)) for diff in pairwise_diffs]
        average = np.mean(total_diff)
        print(total_diff)
        print(average)

number_label = 7
variation = 2
numberOfLayers = 3
multiplierOfN = 2
sample_indices = None

# sample_indices_two_label = [[6780, 5566]]
# sample_indices_two_label_2 = [[6710, 6084], [3465, 296]]

train_uniform_input_per_variation(number_label, variation, numberOfLayers, None, False, None)
# train_uniform_input_per_variation(number_label, variation, None, multiplierOfN, False, None)

# compare_train_uniform_input_per_variation([number_label,2], variation, numberOfLayers, None, [None, None])
# compare_train_uniform_input_per_variation([number_label,4], variation, None, multiplierOfN, [None, None])
