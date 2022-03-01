import numpy as np
from scipy import special
import math
from scipy.optimize import minimize

def cumulative_poissonian(m, l):
    s = 0
    for k in range(m+1):
        s = s + l**k * np.exp(-l) / math.factorial(k)
    return s

def stouffer(z, u):
    z_tot = np.sum(z)/np.sqrt(np.sum(u))
    return z_tot

def significance_to_probability(significance):
    return special.ndtr(significance) - special.ndtr(-significance)

def obj_probability_to_significance(significance, probability):
    return (probability-significance_to_probability(significance))**2

def probability_to_significance(probability):
    res = minimize(obj_probability_to_significance, x0 = 1, args = (probability), method='Nelder-Mead', tol = 1E-6)
    significance = float(res.x)
    return significance

def detection_probability(trigger_number_of_modules, signal_at_distance):
    return (1 - cumulative_poissonian(trigger_number_of_modules - 1, signal_at_distance))

def false_detection_probability(trigger_number_of_modules, background_events):
    return (1 - cumulative_poissonian(trigger_number_of_modules - 2, background_events))
