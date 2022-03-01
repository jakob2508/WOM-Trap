import numpy as np
from scipy.optimize import minimize

class Player:
    def __init__(self):
        self.action = np.random.choice(np.linspace(0, 1, 11))

    @staticmethod
    def _calc_payoff(a):
        if a < 2:
            return (1 - a) * a
        elif a == 4:
            return 0.5 * (1 - a) * a
        else:
            return 0

    def payoff(self, other):
        return self._calc_payoff(self.action, other.action)

    def best_reply(self, other):
        f = lambda x: 1 - self._calc_payoff(x)
        br = minimize(f, 0.5)
        return br.x.item()

A = Player()
B = Player()

print(A.best_reply(B))