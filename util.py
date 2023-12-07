import numpy as np
from brian2 import *
from constant import total_duration, tau
from model_constant_simple import training_mode 


all_alphabets = 'abcdefghijklmnopqrstuvwxyz'
words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

total_operation_counter = 0
test_counter_threshold = 7
time_step = 0.99*tau


def poisson_encoding(image, max_rate):
    operation_counter = 0
    input_rates = np.ceil(image.flatten() / 255.0) * max_rate
    N = len(input_rates) 

    # v is a static variable, because v is updated by the input, thus no need to be leaky
    input_group_excitory = NeuronGroup(N, '''v : 1''', threshold='v>1', reset="v=0")
    input_group_excitory.v = 0

    input_group_inhibtory = NeuronGroup(N, '''v : 1''', threshold='v<1', reset="v=0")
    input_group_inhibtory.v = 0

    @network_operation(dt=time_step)
    def update_v():
        nonlocal operation_counter
        global test_counter_threshold
    
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
            input_group_inhibtory.v[inhibtory_index] = -1.1
        
    return ([input_group_excitory, input_group_inhibtory], 
            [SpikeMonitor(input_group_excitory), SpikeMonitor(input_group_inhibtory)], 
            update_v)

def poisson_encoding_images(images, max_rate):
    global total_operation_counter
    total_operation_counter = 0

    pause_counter = 0
    operation_counter = 0

    len_training_set = len(images)

    input_array = []
    for image in images:
        input_rates = np.ceil(image.flatten() / 255.0) * max_rate
        input_array.append(input_rates)
    
    N = len(input_rates)
    input_pause_threshold = 2
    which_image = 0

    # v is a static variable, because v is updated by the input, thus no need to be leaky
    input_group_excitory = NeuronGroup(N, '''v : 1''', threshold='v>1', reset="v=0")
    input_group_excitory.v = 0

    input_group_inhibtory = NeuronGroup(N, '''v : 1''', threshold='v>1', reset="v=0")
    input_group_inhibtory.v = 0

    @network_operation(dt=time_step)
    def update_v():
        nonlocal input_pause_threshold
        nonlocal which_image
        nonlocal input_array

        global total_operation_counter
        global test_counter_threshold

        nonlocal pause_counter
        nonlocal operation_counter
        
        operation_counter += 1
        input_group_excitory.v = 0 
        input_group_inhibtory.v = 0 

        excitory_indices = np.nonzero(input_array[which_image])[0]
        inhibtory_indices = np.nonzero(input_array[which_image] == 0)[0]

        print(f"total_operation_counter: {total_operation_counter}")
        print(f"len_training_set: {len_training_set}")
        print(f"operation_counter: {operation_counter}")
        print(f"test_counter_threshold: {test_counter_threshold}")
        print(f"pause_counter: {pause_counter}")
        print(operation_counter > test_counter_threshold or total_operation_counter > 15)

        if operation_counter > test_counter_threshold or total_operation_counter > 15:
            # for _ in range(10):
            #     random_index = np.random.randint(0, len(excitory_indices))
            #     excitory_indices[random_index] = 0

            inhibtory_indices = []
            excitory_indices = excitory_indices[:7]

            # testing cue
            # 7 time-step to make sure agg_group.v = 0 (the effect of excitory and inhibtory input wear off to visualize the effect of fully connected agg_group)
            # 3 time-step to test gradual activation of fully connected layer
            if pause_counter > 10:
                pause_counter = 0
                operation_counter = 0
                which_image += 1
                total_operation_counter += 1
                which_image = which_image % len(input_array)
            # clear pause
            elif pause_counter < 7:
                # global training_mode
                # training_mode = 0
                excitory_indices = []
                inhibtory_indices = []
                # only first time-step, we inhibit everything
                if pause_counter == 0:
                    inhibtory_indices = range(len(input_group_inhibtory.v))
                pause_counter += 1
            else:
                pause_counter += 1

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


