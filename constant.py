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

CaIncrement = 0.01
EndoIncrement = 0.01
wIncrement = 0.01

ClMax = 1 
CaMax = 1 # so that ca will decay in reason time in case of a lot of stimulation
EndoMax = 1 # so that Endo will decay in reason time in case of a lot of stimulation
wMax = 0.74 # 1/1.3 which comes from x + 0.3x > 1, because dv/dt retains about 36% of original value after tau

CaMin = EndoMin = ClMin = 0
wMin = 0.0001

wStart = 0.05
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