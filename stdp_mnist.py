from util import *
from brian2 import *
from itertools import combinations
from synapse_layered import simulate_layers
from synapse_agg import simulate_agg
from load_mnist import load_mnist

def load_data():
    return load_mnist()

def train(number_label, variation, max_layers=None, multiplierOfN=None, cross_label=False, sample_indices=None, switch_input=False):
    """Trains the model based on the given parameters."""
    if isinstance(number_label, list):
        input_images = []
        for label in number_label:
            input_image = sample_images(X_separated, label, variation, sample_indices)
            input_images.extend(input_image)
        np.random.shuffle(input_images)
    else:
        input_images = sample_images(X_separated, number_label, variation, sample_indices)


    training_mode = "uniform" if not switch_input else "different"
    print(f"Start {training_mode} input training")

    # Choose simulation function based on multiplierOfN
    simulate = simulate_layers if not multiplierOfN else simulate_agg
    if not switch_input:
        weight_matrices_per_variation = [
            simulate(image, max_layers or multiplierOfN, number_label, index)
            for index, image in enumerate(input_images)
        ]
    else:
        len_images = len(input_images) // 2
        weight_matrices_per_variation = [
            # simulate(input_images[:len_images], max_layers or multiplierOfN, number_label, -1),
            simulate(input_images, max_layers or multiplierOfN, number_label, -1),
            # simulate(input_images[len_images:], max_layers or multiplierOfN, number_label, -2)
        ]

    # # Transpose the weight matrices for cross-label comparison
    # weight_matrices_per_variation = list(map(list, zip(*weight_matrices_per_variation)))

    # # Perform comparison if cross_label is True
    # if not cross_label:
    #     return compare_weights(weight_matrices_per_variation)
    # return weight_matrices_per_variation

def compare_weights(weight_matrices):
    """Compares the weight matrices and prints the differences."""
    for layer in weight_matrices:
        pairwise_diffs = [np.abs(a - b) for a, b in combinations(layer, 2)]
        total_diff = [int(np.sum(diff)) for diff in pairwise_diffs]
        average_diff = np.mean(total_diff)
        print("Differences:", total_diff)
        print("Average difference:", average_diff)

def compare_train(number_labels, variation, max_layers=None, multiplierOfN=None, sample_indices_two_label=None, switch_input=False):
    """Compares the training weights for different labels."""
    weights_per_label = [
        train(label, variation, max_layers, multiplierOfN, True, sample_indices, switch_input)
        for label, sample_indices in zip(number_labels, sample_indices_two_label)
    ]

    label1_weights, label2_weights = weights_per_label

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

# Use the function with specific parameters
X_separated, y_separated = load_data()
number_label = 3
number_label_2 = 8
variation = 20

numberOfLayers = 3
multiplierOfN = 1

sample_indices = None
sample_indices_two_label = [None, None]

# train(number_label, variation, numberOfLayers, None, False, sample_indices, switch_input=False)
# train(number_label, variation, None, multiplierOfN, False, sample_indices, switch_input=False)

# train(number_label, variation, numberOfLayers, None, False, sample_indices, switch_input=True)
# train(number_label, variation, None, multiplierOfN, False, sample_indices, switch_input=True)

train([1,7], variation, None, multiplierOfN, False, sample_indices, switch_input=True)


# compare_train([number_label, number_label_2], variation, numberOfLayers, sample_indices_two_label, False, sample_indices, False)
# compare_train([number_label, number_label_2], variation, None, multiplierOfN, sample_indices_two_label, False)

# compare_train([number_label, number_label_2], variation, numberOfLayers, None, sample_indices_two_label, True)
# compare_train([number_label, number_label_2], variation, None, multiplierOfN, sample_indices_two_label, True)
