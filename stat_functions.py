import numpy as np
from scipy import special
from scipy.optimize import minimize
from scipy.stats import poisson
from scipy.stats import norm

def cumulative_poissonian(k, mu):
    '''
    Calculates the cumulative Poissonian.

    Returns the sum from 0 to k of a Poissonian probability distribution function centered around mu.

    Parameters:
    -----------
    k : interger
    maximum k for until which to sum up to

    mu: float
    mean value of poissonian distribution

    Returns:
    --------
    cumulative_poissonian : float
    cumulative poissonian

    See Also:
    ---------
    scipy.stats.poisson.cdf

    Examples:
    ---------
    >>> cumulative_poissonian(3,5)                      #doctest: +SKIP
    0.2650259152973616
    >>> cumulative_poissonian(0,1)                      #doctest: +SKIP
    1.0
    '''
    dist = poisson(mu)
    cumulative_poissonian = dist.cdf(k)
    return cumulative_poissonian

def stouffer(significances, widths):
    '''
    Combines independent significances.

    Returns the weighted sum of a list of individual significances from independent measurements using Stouffer's method.

    Parameters:
    -----------
    significances : list or array of floats
    list of significances of the independent measurements

    widths: list or array of floats
    list of widths of the independent measurements

    Returns:
    --------
    combined_significance : float
    combined significance

    Examples:
    ---------
    >>> stouffer([10,5],[1,1])                          #doctest: +SKIP
    10.606601717798211
    >>> stouffer([1,4,2],[1,0.5,0.25])                  #doctest: +SKIP
    6.110100926607787
    '''
    combined_significance = np.sum(significances)/np.sqrt(np.sum(np.square(widths)))
    return combined_significance
    

def probability_to_significance(probability, type="two-sided"):
    '''
    Transforms p-value into Z-score.

    Returns the Z-score for a given p-value assuming a Gaussian distribution. One-sided and two-sided tails allowed.

    Parameters:
    -----------
    p-value : float or array of floats

    Returns:
    --------
    Z-score : float or array of floats

    Examples:
    ---------
    >>>  probability_to_significance(0.68)              #doctest: +SKIP
    0.9944578832097535
    >>>  probability_to_significance(0.95)              #doctest: +SKIP
    1.959963984540054          
    >>>  probability_to_significance(0.995)             #doctest: +SKIP
    2.807033768343811
    '''
    if type == "one-sided":
        significance = norm.ppf(probability)
    elif type == "two-sided":
        significance = norm.ppf(probability + (1 - probability)/2)
    return significance

def significance_to_probability(significance, type="two-sided"):
    '''
    Transforms Z-score into p-value.

    Returns the p-value for a given Z-score assuming a Gaussian distribution. One-sided and two-sided tails allowed.

    Parameters:
    -----------
    Z-score : float or array of floats

    Returns:
    --------
    p-value : float or array of floats

    Examples:
    ---------
    >>>  significance_to_probability(1)                 #doctest: +SKIP
    0.6826894921370859
    >>>  significance_to_probability(2)                 #doctest: +SKIP
    0.9544997361036416          
    >>>  significance_to_probability(3)                 #doctest: +SKIP
    0.9973002039367398
    '''
    if type == "one-sided":
        probability = norm.cdf(significance)
    elif type == "two-sided":
        #probability = norm.cdf(significance) - norm.cdf(-significance)
        probability = norm.sf(-significance) - norm.sf(significance)
    return probability

def detection_probability(trigger_number_of_modules, signal_at_distance):
    '''
    Calculates the detection probability of a segmented sensor.

    Returns the probability for trigger_number_of_modules events in the detector to trigger a detection if signal_at_distance signal events are expected from a CCSN at a specific distance.

    Parameters:
    -----------
    trigger_number_of_modules : integer
    number of modules to be hit

    signal_at_distance : float
    extrapolated signal events at a specific distance

    Returns:
    --------
    detection_probability : float
    detection probability

    See Also:
    ---------
    cumulative_poissonian
    false_detection_probability

    Examples:
    ---------
    >>>  detection_probability(10,10)                 #doctest: +SKIP
    0.5420702855281476
    '''
    detection_probability = (1 - cumulative_poissonian(trigger_number_of_modules - 1, signal_at_distance))
    return detection_probability

def false_detection_probability(trigger_number_of_modules, background_events):
    '''
    Calculates the false detection probability of a segmented sensor.

    Returns the probability for trigger_number_of_modules events in the detector to originate from backhround if one expects background_events background events.

    Parameters:
    -----------
    trigger_number_of_modules : integer
    number of modules to be hit

    background_events : float
    background events

    Returns:
    --------
    false_detection_probability : float
    false detection probability

    See Also:
    ---------
    cumulative_poissonian
    detection_probability

    Examples:
    ---------
    >>>  detection_probability(10,5)                 #doctest: +SKIP
    0.06809363472184837
    '''
    false_detection_probability = (1 - cumulative_poissonian(trigger_number_of_modules - 2, background_events))
    return false_detection_probability
