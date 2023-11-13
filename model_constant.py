# -(theta+fire_threshold/1.5) pushes theta to go to -fire_threshold/1.5 rather than 0, introduces homestatsis
neuron_eqs = '''
dv/dt = -v/tau : 1
dtheta/dt = (-(theta+fire_threshold/1.5))/tau : 1
'''

# neuron_eqs = '''
# v:1
# dtheta/dt = -theta/tau : 1
# '''

threshold_eqs = 'v > fire_threshold + theta'

reset_eqs = '''
theta += (v-fire_threshold-theta) thetaIncrement
theta = clip(theta, 0, theta)
v = 0
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
