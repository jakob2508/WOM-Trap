from copyreg import pickle
import globals
from sensors import *
from scipy.optimize import brute
from plthelper import *
import pickle

debug1 = 0 #plot grid scan as colour plot or contour
debug2 = 1 #plot detection horizon and probability as function of distance for optimized trigger condition

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
    type = "WLS" #"AND", "OR", "MDOM", "WLS"

    if type == "MDOM":
        optimise = MDOM_optimise
        DETECTOR = MDOM
    elif type == "WLS":
        DETECTOR = WLS
    elif type == "AND":
        optimise = AND_optimise
        DETECTOR = AND
    elif type == "OR":
        optimise = OR_optimise
        DETECTOR = OR

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

    data = pmt_range, modules_range, detection_horizon, false_alarm_rate
    file = open('data_' + type + '.pkl', 'wb')
    pickle.dump(data, file)
    file.close()
    a = 1/0
    '''
    if debug1 == 1 and type != "WLS":
        file = open('data_' + type + '.pkl', 'rb')
        data = pickle.load(file)
        pmt_range, modules_range, detection_horizon, false_alarm_rate = data
        file.close()

        fig, ax = grid_plot(pmt_range, modules_range, detection_horizon, false_alarm_rate)
        plt.savefig('./plots/grid_scan_'+type+'.png') 
        plt.show()  
        plt.close(fig)

        fig, ax = contour_plot(pmt_range, modules_range, detection_horizon, false_alarm_rate)
        plt.savefig('./plots/contour_'+type+'.png') 
        plt.show()  
        plt.close(fig)        

    '''
    slices = (slice(4,9,1), slice(2,120,1))
    res = brute(optimise, ranges = slices, disp=True, finish=None, full_output=True)
    opt_trigger_number_of_PMTs, opt_trigger_number_of_modules = res[0].astype(int)
    '''
    if type == 'MDOM':
        opt_trigger_number_of_PMTs, opt_trigger_number_of_modules = 7, 7
        detector = DETECTOR('mDOM', opt_trigger_number_of_PMTs, opt_trigger_number_of_modules)
    elif type == 'WLS':
        opt_trigger_number_of_PMTs, opt_trigger_number_of_modules = np.nan, np.nan
        detector = DETECTOR('wls')
    else:
        if type == 'AND':
            opt_trigger_number_of_PMTs, opt_trigger_number_of_modules = 6, 19
        elif type == 'OR':
            opt_trigger_number_of_PMTs, opt_trigger_number_of_modules = 7, 7
        detector = DETECTOR('mDOM', 'wls', opt_trigger_number_of_PMTs, opt_trigger_number_of_modules)
        mdom = MDOM('mDOM', opt_trigger_number_of_PMTs, opt_trigger_number_of_modules)
        wls = WLS('WLS')
    
    print('Optimised trigger condition for '+type+' type: [{:.0f}, {:.0f}]'.format(opt_trigger_number_of_PMTs, opt_trigger_number_of_modules))
    print('-----------------------------------------------------')
    print('False SN alarm rate: {:.4f}/year --> 1 false alarm every {:.1f} years'.format(detector.false_alarm_rate(), 1/detector.false_alarm_rate()))
    print('50% SN detection horizon: {:.1f} kpc'.format(detector.detection_horizon()))

    if debug2:
        if (type == "AND") or (type == "OR"):
            fig, axs = result_plot(mdom, wls, detector)
        elif (type == "MDOM") or (type == "WLS"):
            fig, axs = result_plot_single(type, detector)

        plt.tight_layout()
        plt.savefig('./plots/opt_distance_'+type+'.png')
        plt.show()


'''
RESULTS:

Optimised trigger condition for MDOM type: [7, 7]
-----------------------------------------------------
False SN alarm rate: 0.0103/year --> 1 false alarm every 97.5 years
50% SN detection horizon: 268.3 kpc

Optimised trigger condition for AND type: [6, 19]
-----------------------------------------------------
False SN alarm rate: 0.0033/year --> 1 false alarm every 303.2 years
50% SN detection horizon: 192.6 kpc

Optimised trigger condition for OR type: [7, 7]
-----------------------------------------------------
False SN alarm rate: 0.0103/year --> 1 false alarm every 97.5 years
50% SN detection horizon: 274.4 kpc

'''