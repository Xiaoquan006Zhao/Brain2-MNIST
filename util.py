import numpy as np
from brian2 import *
from constant import total_duration, tau


all_alphabets = 'abcdefghijklmnopqrstuvwxyz'
words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']


def poisson_encoding(image, max_rate):
    input_rates = np.ceil(image.flatten() / 255.0) * max_rate
    N = len(input_rates) 

    # time_step = 0.999*tau
    # high-frequency stimulation
    time_step = 0.99*tau

    # v is a static variable, because v is updated by the input, thus no need to be leaky
    input_group_excitory = NeuronGroup(N, '''v : 1''', threshold='v>1', reset="v=0")
    input_group_excitory.v = 0

    input_group_inhibtory = NeuronGroup(N, '''v : 1''', threshold='v>1', reset="v=0")
    input_group_inhibtory.v = 0

    operation_counter = 0
    test_counter_threshold = 30

    @network_operation(dt=time_step)
    def update_v():
        nonlocal operation_counter
        # nonlocal excitory_indices
        # nonlocal inhibtory_indices
        operation_counter += 1

        excitory_indices = np.nonzero(input_rates)[0]
        inhibtory_indices = np.nonzero(input_rates == 0)[0]

        input_group_excitory.v = 0 
        input_group_inhibtory.v = 0
        
        # cover up half of the pixels and test memory recall
        if operation_counter > test_counter_threshold:
            operation_counter = 0
            # for _ in range(5):
            #     random_index = np.random.randint(0, len(excitory_indices))
            #     excitory_indices[random_index] = 0
            excitory_indices = excitory_indices[:int(len(excitory_indices)/2)]
        
        for excitory_index in excitory_indices:
            input_group_excitory.v[excitory_index] = 1.1
        
        for inhibtory_index in inhibtory_indices:
            input_group_inhibtory.v[inhibtory_index] = 1.1
        
    return ([input_group_excitory, input_group_inhibtory], 
            [SpikeMonitor(input_group_excitory), SpikeMonitor(input_group_inhibtory)], 
            update_v)

def poisson_encoding_images(images, max_rate):
    input_array = []
    for image in images:
        input_rates = np.ceil(image.flatten() / 255.0) * max_rate
        input_array.append(input_rates)
    
    N = len(input_rates)
    operation_counter = 0
    input_switch_threshold = 30
    input_pause_threshold = 2

    # time_step = 0.999*tau
    # high-frequency stimulation
    time_step = 0.99*tau

    which_image = 0

    # v is a static variable, because v is updated by the input, thus no need to be leaky
    input_group_excitory = NeuronGroup(N, '''v : 1''', threshold='v>1', reset="v=0")
    input_group_excitory.v = 0

    input_group_inhibtory = NeuronGroup(N, '''v : 1''', threshold='v>1', reset="v=0")
    input_group_inhibtory.v = 0

    @network_operation(dt=time_step)
    def update_v():
        nonlocal input_pause_threshold
        nonlocal input_switch_threshold
        nonlocal operation_counter
        nonlocal which_image
        nonlocal input_array

        operation_counter += 1
        input_group_excitory.v = 0 
        input_group_inhibtory.v = 0 

        excitory_indices = np.nonzero(input_array[which_image])[0]
        inhibtory_indices = np.nonzero(input_array[which_image] == 0)[0]

        if operation_counter > input_switch_threshold:
            operation_counter = 0
            which_image += 1
            which_image = which_image % len(input_array)

            # for _ in range(5):
            #     random_index = np.random.randint(0, len(excitory_indices))
            #     excitory_indices[random_index] = 0
            excitory_indices = excitory_indices[:int(len(excitory_indices)/2)]
            
        
        # if operation_counter > input_pause_threshold:
        for excitory_index in excitory_indices:
            input_group_excitory.v[excitory_index] = 1.1
    
        for inhibtory_index in inhibtory_indices:
            input_group_inhibtory.v[inhibtory_index] = 1.1

    return ([input_group_excitory, input_group_inhibtory], 
            [SpikeMonitor(input_group_excitory), SpikeMonitor(input_group_inhibtory)], 
            update_v)


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

def perfect_square(number):
    # Check if the number is already a perfect square
    if (number**0.5).is_integer():
        return (int(number**0.5), int(number**0.5))

    # If not a perfect square, find the closest combination
    for i in range(int(number**0.5), 0, -1):
        if number % i == 0:
            return (number // i, i)

    # If no exact divisor found, return the closest combination
    for i in range(int(number**0.5), 0, -1):
        j = number // i
        if i * j < number:
            return (j + 1, i)

    return None


