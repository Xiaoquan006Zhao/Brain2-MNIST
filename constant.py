from brian2 import *

seed(427)

tau = 5*ms
tau_Ca = 0.8*tau
tau_Endo = 0.6*tau

taupre = taupost = 0.7*tau
Apre = 0.01
Apost = Apre*taupre/taupost*0.95

fire_threshold = 1
thetaIncrement = 0.005

neuron_eqs = '''
dv/dt = -v/tau : 1
dtheta/dt = -theta/(2*tau) : 1
'''

# neuron_eqs = '''
# v:1
# dtheta/dt = -theta/(2*tau) : 1
# '''

threshold_eqs = 'v > fire_threshold + theta'

reset_eqs = '''
v = 0
theta += thetaIncrement
'''

synapse_voltage_model = '''
w : 1
'''

synapse_learning_model = '''
dCa/dt = -Ca/tau_Ca : 1 (event-driven)
dapre/dt = -apre/taupre : 1 (event-driven)
dapost/dt = -apost/taupost : 1 (event-driven)
'''

# v_post += w + condition_high_voltage*v_post*0.1, make it easier to fire when close to fire
    # like opening more voltage gates when close to fire
# (100 * (Ca-0.2) * (0.4-Ca)) meanings center at 0.3, just to scale the LTD 
# (Ca - 0.7) * 3, same reason as above
on_pre_model = '''
Ca_temp = Ca
v_temp = v_post

condition_high_voltage = int(v_temp > 0.75)
v_temp += w + condition_high_voltage*v_post*0.1

Ca_temp = clip(Ca_temp + w, CaMin, CaMax)

condition_Remove_Mg = int(v_temp > 0.7)
condition_LTD = int(Ca_temp > 0.2 and Ca_temp < 0.4)
condition_LTP = int(Ca_temp > 0.7)

w -= wIncrement * condition_Remove_Mg * condition_LTD * (100 * (Ca_temp-0.2) * (0.4-Ca_temp))
w += wIncrement * condition_Remove_Mg * condition_LTP * (Ca_temp - 0.7) * 3

w -= apost
w = clip(w, wMin, wMax)

apre += Apre
v_post = v_temp
Ca = Ca_temp
'''

# v_pre += w + condition_high_voltage*v_post*0.1
on_post_model = '''
w += apre
w = clip(w, wMin, wMax)

condition_high_voltage = int(v_pre > 0.75)


apost += Apost
'''

CaIncrement = 0.01
EndoIncrement = 0.01
wIncrement = 0.01

ClMax = 1 
CaMax = 1 # so that ca will decay in reason time in case of a lot of stimulation
EndoMax = 1 # so that Endo will decay in reason time in case of a lot of stimulation
wMax = 0.74 # 1/1.3 which comes from x + 0.3x > 1, because dv/dt retains about 36% of original value after tau

CaMin = EndoMin = ClMin = 0
wMin = 0.0001

wStart = 0.0009
wAddThreshold = 0.6

max_rate = 300 * Hz
iteration = 3
duration = 50 * ms
total_duration = iteration * duration
total_duration_graph = total_duration/ms
# interval = total_duration_graph / 6

interval = tau/ms

input_connect_probability = 0.003
agg_connect_probability = 0.0005