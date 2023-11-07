import numpy as np
from brian2 import *
from constant import total_duration, tau


all_alphabets = 'abcdefghijklmnopqrstuvwxyz'
words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

def poisson_encoding(image, max_rate):
    input_rates = np.ceil(image.flatten() / 255.0) * max_rate
    N = len(input_rates) 

    time_step = 0.99*tau

    input_indices = np.nonzero(input_rates)[0]

    eqsWInput = '''
        v : 1
    '''

    input_group = NeuronGroup(N, eqsWInput, threshold='v>1', reset="v=0")
    input_group.v = 0

    @network_operation(dt=time_step)
    def update_v():
        input_group.v = 0 
        for input_index in input_indices:
            input_group.v[input_index] = 1.1

    return (input_group, SpikeMonitor(input_group), update_v)

def sample_images(image_database, label, num_samples, indices):
    if label < 0 or label > 9:
        raise ValueError(f"Label ({label}) is not supported).")
    
    images_of_label = image_database[label]
    if num_samples > len(images_of_label):
        raise ValueError(f"Number of samples requested ({num_samples}) is greater than the available samples ({len(images_of_label)}).")
    
    if not indices:
        # seed(None)
        random_indices = np.random.choice(len(images_of_label), num_samples, replace=False)
        # seed(brian_seed)
    else:
        random_indices = indices

    print(f"{label} label, variation {random_indices}")

    images = images_of_label[random_indices]
    images_augmented = []

    one_hot_encoded_word = one_hot_encode_word(words[label])
    one_hot_encoded_word = [300] + one_hot_encoded_word + [300]  # make dimension match MNIST

    for image in images:
        image = np.insert(image, 0, one_hot_encoded_word, axis=0)
        image = image.reshape(29, 28)
        images_augmented.append(image)

    

    return images_augmented

def one_hot_encode_word(word):
    
    char_to_index = {char: index for index, char in enumerate(all_alphabets)}
    
    one_hot_vector = [0] * len(all_alphabets)
        
    for char in word:
        one_hot_vector[char_to_index[char]] = 300

    return one_hot_vector


