import numpy as np
from datasheet import *
from stat_functions import *

class Sensor(object):
    def __init__(self, name, number_of_PMTs, signal_events, noise_rate):
        self.name = name
        self.number_of_PMTs = number_of_PMTs
        self.signal_events = signal_events
        self.noise_rate = noise_rate

    def background_events(self):
        return self.noise_rate * dT_SN

    def signal_at_distance(self, distance, distance_0 = distance_0):
        return self.signal_events * (distance_0/distance)**2

    def significance(self, distance, type = "poisson"):
        s = self.signal_at_distance(distance)
        b = self.background_events()
        n = s + b
        if type == "poisson":
            Z = np.sqrt(2*(n*np.log(n/b)-(n-b)))
        elif type == "gauss":
            Z = np.abs(n-s)/np.sqrt(b)
        return Z

class MDOM(Sensor):
    def __init__(self, name, number_of_PMTs, trigger_number_of_modules, signal_events, noise_rate):
        super().__init__(name, number_of_PMTs, signal_events, noise_rate)
        self.trigger_number_of_modules = trigger_number_of_modules

    @staticmethod
    def _detection_probability(self, distance):
        return detection_probability(self.trigger_number_of_modules, self.signal_at_distance(distance))

    def false_detection_probability(self):
        return false_detection_probability(self.trigger_number_of_modules, self.background_events())

    def false_alarm_rate(self):
        return self.false_detection_probability()*year*self.noise_rate

    # def obj_detection_horizon(self, distance):
    #     return (self.detection_probability(distance) - 0.5)**2

    #     f = lambda x: (detection_probability(x)-0.5)**2

    def detection_horizon(self):   
        f = lambda x: (self._detection_probability(self,x)-0.5)**2
        res = minimize(f, x0 = 100, method='Nelder-Mead', tol = 1E-6)
        return res.x.item()