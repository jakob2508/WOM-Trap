from sensors import *

#### LOGICAL AND ####

def AND_detection_probability(distance):
    return mDOM.detection_probability(distance) * wls.detection_probability(distance)

def AND_false_detection_probability():
    return mDOM.false_detection_probability() * wls.false_detection_probability()

def AND_significance(distance):
    AND_detection_probability = mDOM.detection_probability(distance)*wls.detection_probability(distance)
    return probability_to_significance(AND_detection_probability)

def AND_detection_horizon():
    thresh = 1E-3
    f = lambda x: ((mDOM.detection_probability(x)*wls.detection_probability(x))-0.5)**2
    loss = 1
    x0 = 1
    while loss > thresh:
        res = minimize(f, x0 = x0, method='Nelder-Mead', tol = 1E-6)
        loss = (AND_detection_probability(res.x.item())-0.5)**2
        x0 += 50
        if x0 > 1E4:
            break
    return res.x.item()

def AND_noise_rate():
    return mDOM.noise_rate * wls.noise_rate

def AND_false_alarm_rate():
    return AND_false_detection_probability() * AND_noise_rate() * year

mDOM = MDOM('mDOM', 7, 7)
wls = WLS('WLS')

'''
def AND(trigger_number_of_PMTs, trigger_number_of_modules,distance):
    mDOM = MDOM('mDOM', trigger_number_of_PMTs, trigger_number_of_modules)
    wls = WLS('WLS')
    AND_noise_rate = mDOM.noise_rate * wls.noise_rate
    AND_false_detection_probability = mDOM.false_detection_probability()*wls.false_detection_probability()
    AND_detection_probability = AND_detection_probability(distance)
    AND_false_alarm_rate = AND_false_detection_probability*AND_noise_rate*year
    AND_significance = AND_significance(distance)
    AND_detection_horizon = AND_detection_horizon()
    f = lambda x : 

    return AND_false_alarm_rate, AND_detection_horizon


#### LOGICAL OR ####
def OR(trigger_number_of_PMTs, trigger_number_of_modules):
    mDOM = MDOM('mDOM', trigger_number_of_PMTs, trigger_number_of_modules)
    wls = WLS('WLS')
    OR_noise_rate = mDOM.noise_rate + wls.noise_rate
    OR_significance = stouffers([mDOM.signifance, wls.significance])
    OR_detection_probability = significance_to_probability(OR_significance)
    OR_false_detection_probability = 1 - OR_detection_probability
    OR_false_alarm_rate = OR_false_detection_probability*OR_noise_rate*year
'''