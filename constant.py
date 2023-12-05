from brian2 import *

seed(437)

tau = 5*ms
tau_Ca = 0.8*tau
tau_Endo = 0.6*tau

taupre = taupost = 0.7*tau
Apre = 0.01
Apost = Apre*taupre/taupost*0.95

fire_threshold = 1
thetaIncrement = 0.005

CaIncrement = 0.01
EndoIncrement = 0.01
wIncrement = 0.1

ClMax = 1 
CaMax = 1 # so that ca will decay in reason time in case of a lot of stimulation
EndoMax = 1 # so that Endo will decay in reason time in case of a lot of stimulation
# wMax = 0.74 # 1/1.3 which comes from x + 0.3x > 1, because dv/dt retains about 36% of original value after tau
wMax = 1

CaMin = EndoMin = ClMin = 0
wMin = -1

wStart = 0.1
wAddThreshold = 0.6

max_rate = 300 * Hz

iteration = 80
duration = 5 * ms
total_duration = iteration * duration
total_duration_graph = total_duration/ms

# interval = total_duration_graph / 6

interval = tau/ms

input_connect_probability = 0
agg_connect_probability = 1