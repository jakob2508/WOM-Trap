from cProfile import label
import globals
from sensors import *
from scipy.optimize import brute
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def grid_plot(x,y,detection_horizon, false_alarm_rate):

    loss = np.sqrt(1/detection_horizon**2 + globals.fit_weight_OR * (globals.DES_false_alarm_rate_comb - false_alarm_rate)**2)

    mask = np.where(loss == loss.min())

    fig, ax = plt.subplots(1,2, figsize = (15,5))

    xx, yy = np.meshgrid(x,y)
    clp = ax[0].scatter(xx, yy, c = detection_horizon.T, s = 10, marker = 'o', alpha = 0.5, cmap = 'viridis', norm=LogNorm())
    ax[0].plot(xx[mask], yy[mask], detection_horizon.T[mask], s = 10, marker = 'o', color = 'red')
    ax[0].set_xlabel(r'$n_{trigger}$')
    ax[0].set_ylabel(r'$N_{\nu}$')
    ax[0].set_xticks(x)
    clb = plt.colorbar(clp, ax=ax[0])
    clb.set_label('detection horizon [kpc]', fontsize = 10)
    clp = ax[1].scatter(xx, yy, c = 100*false_alarm_rate.T, s = 10, marker = 'o', alpha = 0.5, cmap = 'viridis', norm=LogNorm())
    ax[1].plot(xx[mask], yy[mask], false_alarm_rate.T[mask], s = 10, marker = 'o', color = 'red')
    ax[1].set_xlabel(r'$n_{trigger}$')
    ax[1].set_ylabel(r'$N_{\nu}$')
    ax[1].set_xticks(x)
    clb = plt.colorbar(clp, ax=ax[1])
    clb.set_label('false alarm rate [1/century]', fontsize = 10)

    return fig, ax



def AND_optimise(input):
    print(input)
    trigger_number_of_PMTs, trigger_number_of_modules = input
    land = AND('mDOM', 'wls', trigger_number_of_PMTs, trigger_number_of_modules)

    return np.sqrt((1/land.detection_horizon())**2 + globals.fit_weight_AND * (land.false_alarm_rate() - globals.DES_false_alarm_rate_comb)**2)

def OR_optimise(input):
    print(input)
    trigger_number_of_PMTs, trigger_number_of_modules = input
    lor = OR('mDOM', 'wls', trigger_number_of_PMTs, trigger_number_of_modules)

    return np.sqrt((1/lor.detection_horizon())**2 + globals.fit_weight_OR * (lor.false_alarm_rate() - globals.DES_false_alarm_rate_comb)**2)

def MDOM_optimise(input):
    print(input)
    trigger_number_of_PMTs, trigger_number_of_modules = input
    mDOM = MDOM('mDOM', trigger_number_of_PMTs, trigger_number_of_modules)

    return np.sqrt((1/mDOM.detection_horizon())**2 + globals.fit_weight_mDOM * (mDOM.false_alarm_rate() - globals.DES_false_alarm_rate_mDOM)**2)

if __name__ == "__main__": 
    globals.initialize()
    type = "AND" # "AND", "OR", "MDOM"

    if type == "AND":
        optimise = AND_optimise
        DETECTOR = AND
    elif type == "OR":
        optimise = OR_optimise
        DETECTOR = OR
    elif type == "MDOM":
        optimise = MDOM_optimise
        DETECTOR = MDOM
    '''
    pmt_range = np.arange(4,9,1)
    modules_range = np.arange(2,120,1)
    detection_horizon = np.zeros((len(pmt_range), len(modules_range)))
    false_alarm_rate = np.zeros((len(pmt_range), len(modules_range)))
    for i, n_pmt in enumerate(pmt_range):
        for j, n_mod in enumerate(modules_range):
            print(n_pmt, n_mod)
            detector = DETECTOR('mDOM', 'wls', n_pmt, n_mod)
            #detector = DETECTOR('mDOM', n_pmt, n_mod)
            detection_horizon[i,j] = detector.detection_horizon()
            false_alarm_rate[i,j] = detector.false_alarm_rate()

    fig, ax = grid_plot(pmt_range, modules_range, detection_horizon, false_alarm_rate)
    plt.savefig('grid_scan_'+type+'.png') 
    plt.show()  
    plt.close(fig)        
    '''

    
    slices = (slice(4,9,1), slice(2,120,1))
    res = brute(optimise, ranges = slices, disp=True, finish=None, full_output=True)
    opt_trigger_number_of_PMTs, opt_trigger_number_of_modules = res[0].astype(int)
    
    '''
    if type == 'AND':
        opt_trigger_number_of_PMTs, opt_trigger_number_of_modules = 6, 19
    elif type == 'OR':
        opt_trigger_number_of_PMTs, opt_trigger_number_of_modules = 6, 17
    '''
    detector = DETECTOR('mDOM', 'wls', opt_trigger_number_of_PMTs, opt_trigger_number_of_modules)

    print('Optimised trigger condition for '+type+' type: [{:.0f}, {:.0f}]'.format(opt_trigger_number_of_PMTs, opt_trigger_number_of_modules))
    print('-----------------------------------------------------')
    print('False SN alarm rate: {:.4f}/year --> 1 false alarm every {:.1f} years'.format(detector.false_alarm_rate(), 1/detector.false_alarm_rate()))
    print('50% SN detection horizon: {:.1f} kpc'.format(detector.detection_horizon()))

    fig, axs = plt.subplots(1,2, figsize = (8,4))
    ax1 = axs[0]
    ax2 = axs[1]

    distance = np.linspace(5, 1500, 1000)

    Z_mDOM = MDOM('mDOM', opt_trigger_number_of_PMTs, opt_trigger_number_of_modules).significance(distance)
    Z_WLS = WLS('WOM-Trap').significance(distance)

    P_SN_mDOM = MDOM('mDOM', opt_trigger_number_of_PMTs, opt_trigger_number_of_modules).detection_probability(distance)
    P_SN_WLS = WLS('WOM-Trap').detection_probability(distance)

    P_SN_comb = np.zeros_like(distance)
    Z_comb = np.zeros_like(distance)
    for i,dd in enumerate(distance):
        P_SN_comb[i] = detector.detection_probability(dd)
        Z_comb[i] = detector.significance(dd)

    ax1.plot(distance, Z_mDOM, label = "mDOM", color = 'grey', ls = ':')
    ax1.plot(distance, Z_WLS, label = "WLS", color = 'grey', ls = '--')
    ax1.plot(distance, Z_comb, label = "WOM-Trap", color = 'black', ls = '-')
    ax2.plot(distance, P_SN_mDOM, label = "mDOM", color = 'grey', ls = ':')
    ax2.plot(distance, P_SN_WLS, label = "WLS", color = 'grey', ls = '--')
    ax2.plot(distance, P_SN_comb, label = "WOM-Trap", color = 'black', ls = '-')

    ax1.set_xlabel("Distance d [kpc]")
    ax1.set_ylabel("Significance Z")
    ax1.set_yscale("log")
    ax2.set_xlabel("Distance d [kpc]")
    ax2.set_ylabel(r"SN detection probability $P_{SN}$")
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.savefig('opt_distance_'+type+'.png')
    plt.show()

'''
RESULTS:

Optimised trigger condition for AND type: [6, 19]
-----------------------------------------------------
False SN alarm rate: 0.0033/year --> 1 false alarm every 303.2 years
50% SN detection horizon: 192.6 kpc

Optimised trigger condition for OR type: [8, 7]
-----------------------------------------------------
False SN alarm rate: 0.0100/year --> 1 false alarm every 100.0 years
50% SN detection horizon: 1111.7 kpc

'''