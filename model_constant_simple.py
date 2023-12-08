from constant import Apre

# -(theta+fire_threshold/1.5) pushes theta to go to -fire_threshold/1.5 rather than 0, introduces homestatsis
neuron_eqs = '''
dv/dt = -v/tau : 1
dtheta/dt = (-(theta+fire_threshold/1.5))/tau : 1
'''

# neuron_eqs = '''
# v:1
# dtheta/dt = -theta/tau : 1
# '''

threshold_eqs = 'v > fire_threshold'

reset_eqs = '''
v = 0
'''

synapse_voltage_model = '''
w : 1
'''

synapse_learning_model = '''
dapre/dt = -apre/taupre : 1 (event-driven)
dapost/dt = -apost/taupost : 1 (event-driven)
'''

# nearly no activity
no_activation_threshold = 0.001

# recent 1 activity because Apre = 0.01
yes_activation_threshold = Apre*0.3
training_flag = 1

# no activation of post in recent activities, thus decrease the weight
on_pre_model = '''
apre += Apre

condition_no_post_activation = int(apost < no_activation_threshold)
w -= condition_no_post_activation * wIncrement 

condition_yes_post_activation = int(apost > yes_activation_threshold)
w += condition_yes_post_activation * wIncrement 

w = clip(w, wMin, wMax)
v_post += w**5
'''

# no activation of post in recent activities, thus increase the weight (because of inhibtory)
on_pre_model_inhibtory = '''
apre += Apre

condition_no_post_activation = int(apost < 0.01)
w += condition_no_post_activation * wIncrement

w = clip(w, wMin, wMax)
v_post += w
'''

# no activation of pre in recent activities, thus decrease the weight
on_post_model = '''
apost += Apost

condition_no_pre_activation = int(apre < no_activation_threshold)
w -= condition_no_pre_activation * wIncrement 

condition_yes_pre_activation = int(apre > yes_activation_threshold)
w += condition_yes_pre_activation * wIncrement 

w = clip(w, wMin, wMax)
'''

# no activation of pre in recent activities, thus increase the weight (because of inhibtory)
on_post_model_inhibtory = '''
apost += Apost

condition_no_pre_activation = int(apre < 0.01)
w -= condition_no_pre_activation * wIncrement

condition_yes_pre_activation = int(apre > 0.03)
w += condition_yes_pre_activation * wIncrement

w = clip(w, wMin, wMax)
'''
