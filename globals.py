def initialize(): 
    global fit_weight_mDOM
    global fit_weight_AND
    global fit_weight_OR
    global DES_false_alarm_rate_comb
    global DES_false_alarm_rate_mDOM
    global DES_false_alarm_rate_WLS

    fit_weight_mDOM = 1
    fit_weight_AND = 1E-2
    fit_weight_OR = 1E-1
    DES_false_alarm_rate_comb = 0.01 # desired combined false alarm rate
    DES_false_alarm_rate_mDOM = 0.01 # desired mDOM false alarm rate
    DES_false_alarm_rate_WLS = 0.01 # desired WLS false alarm rate