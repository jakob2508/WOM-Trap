from cProfile import label
from datasheet import *
from plthelper import *
import numpy as np
from scipy.stats import poisson, norm, chi2
from scipy.special import gamma
from scipy.optimize import minimize
from iminuit import Minuit
import matplotlib.pyplot as plt

debug1 = 1 # p-value and Z-score for different coincidence conditions at fixed distance
debug2 = 1 # detection horizon for different coincidence conditions at N sigma level

def cont_poisson(k, mu):
    return np.exp(-mu) * mu**k * 1/gamma(k+1)

def get_LH(signal_10kpc, distance, detector_type, coinc = None, llh_type = "Asimov"):

    signal = signal_10kpc * (10/distance)**2

    if detector_type == "DOM":
        mean = signal + f_noise_DOM * dT_SN
        if llh_type == "Asimov":
            n_obs = signal_DOM*(10/distance)**2 + f_noise_DOM * dT_SN
            return norm.pdf(n_obs, loc = mean, scale = np.sqrt(n_obs))

    if detector_type == "WLS":
        mean = signal * ratio_WLS + f_noise_WLS * dT_SN
        if llh_type == "Asimov":
            n_obs = signal_WLS*(10/distance)**2 + f_noise_WLS * dT_SN
            return norm.pdf(n_obs, loc = mean, scale = np.sqrt(n_obs))

    if detector_type == "mDOM":
        mean = signal * ratio_mDOM[coinc-1] + f_noise_mDOM[coinc-1] * dT_SN
        if llh_type == "Asimov":
            n_obs = signal_mDOM[coinc-1]*(10/distance)**2 + f_noise_mDOM[coinc-1] * dT_SN
            if n_obs < 100:
                return cont_poisson(n_obs, mean)
            else:
                return norm.pdf(n_obs, loc = mean, scale = np.sqrt(n_obs))

def get_comb_LLH(signal, distance, coinc):
    LH_DOM = get_LH(signal, distance, "DOM")
    LH_WLS = get_LH(signal, distance, "WLS")
    LH_mDOM = get_LH(signal, distance, "mDOM", coinc = coinc)
    likelihoods = np.array([LH_DOM, LH_WLS, LH_mDOM])
    return np.sum(np.log(likelihoods))

def get_max_LLH(distance, coinc):
    sig_true = 1E6
    loss_true = -get_comb_LLH(sig_true, distance, coinc) 
    x0 = (sig_true-1000)
    flag, brk = True, False
    while flag:
        opt = Minuit(lambda x: -get_comb_LLH(x, distance, coinc), x0)
        opt.tol = 1E-3
        opt.errors = 1E-1
        opt.errordef=Minuit.LEAST_SQUARES
        opt.limits = [(0, None)]
        opt.strategy=2
        opt.simplex()
        opt.migrad()
        opt.migrad()

        if np.abs(opt.fval <= loss_true) and opt.values[0] != sig_true:
            flag = False 
        # if np.abs(opt.values[0]-sig_true) > 1E0:
        else:
            x0 += 100
            if x0 >= (1E6+1000):
                print("WARNING: Minimizer reached end of start parameter range")
                brk = True
        # else:
        #     flag = False
        if brk:
            break

    return -opt.fval, opt.values[0]

def get_TS(distance, coinc):
    max_sig_llh, max_sig = get_max_LLH(distance, coinc)
    null_llh = get_comb_LLH(0, distance, coinc)
    return -2*(null_llh - max_sig_llh)

def get_PV(distance, coinc):
    TS = get_TS(distance, coinc)
    PV = 1 - chi2.cdf(TS, 1)
    return PV
    
def get_Z(distance, coinc):
    PV = get_PV(distance, coinc)
    Z = norm.ppf((1-PV) + PV/2)
    return Z

# def chi2(signal_range, detector_type, coinc = None):
#     H1 = get_LH(signal_range, distance = 100, detector_type = detector_type, coincidence = coinc)
#     H0 = get_LH(0, distance = 100, detector_type = detector_type, coincidence = coinc)
#     chi2 = -2*np.log(H1/H0)
#     return chi2


def loss_distance(distance, coinc, z_ref):
    z = get_Z(distance, coinc)
    loss = np.sqrt((z-z_ref)**2)
    return loss

def get_distance(coinc, z_ref):
    x0 = 500
    opt = Minuit(lambda x: loss_distance(x, coinc, z_ref), (x0), name = ('d'))
    opt.tol = 1E-2
    opt.errors = (1E-1)
    opt.errordef=Minuit.LEAST_SQUARES
    opt.limits = [(0, None)]
    opt.strategy=2
    opt.simplex()
    opt.migrad()
    return opt.values[0]

def plot_distance(coinc):
    PV, Z = [], []
    dist_z_ref = get_distance(coinc, z_ref)
    distance_range = np.linspace(dist_z_ref - 100, dist_z_ref + 100, 100)
    for distance in distance_range:
        PV.append(get_PV(distance, coinc))
        Z.append(get_Z(distance, coinc))
    
    PV, Z = np.array(PV), np.array(Z)

    fig, ax = plt.subplots(1,1)
    ax2 = ax.twinx()
    ax.plot(distance_range, PV, 'x', color = "C0")
    ax2.plot(distance_range, Z, 'o', color = "C1")
    ax.set_xlabel('distance [kpc]')
    ax.set_ylabel('p-value')
    ax2.set_ylabel('Z-score')
    ax2.axvline(dist_z_ref, color = 'k')
    ax.yaxis.label.set_color("C0")
    ax2.yaxis.label.set_color("C1")
    ax.tick_params(axis='y', colors="C0")
    ax2.tick_params(axis='y', colors="C1")
    ax2.text(dist_z_ref, 6, s = r"5$\sigma$", ha = "center", va = "center", color = 'k', bbox = dict(boxstyle="square", facecolor = "white"))
    #ax2.set_yscale('log')
    plt.title("p-value/Z-score over distance for {:.0f}-fold coincidence".format(coinc))
    plt.savefig("./plots/{:.0f}sigma_horizon_coinc=".format(z_ref)+str(coinc)+".png")
    plt.show()

# consider 100kpc distance for now as 10 kpc results in a too large deviation in IceCube and WLS case whihc results in very small probabilities, which in turn results in vanishing likelihoods for the null hypothesis

SN_type = "heavy"
update_type = "Gen2"

n_PMT_DOM, signal_DOM, f_noise_DOM = allocator("DOM", update_type, SN_type, None)
n_PMT_WLS, signal_WLS, f_noise_WLS = allocator("WLS", update_type, SN_type, None)
n_PMT_mDOM, signal_mDOM, f_noise_mDOM = allocator("mDOM", update_type, SN_type, None)

ratio_WLS = signal_WLS/signal_DOM
ratio_mDOM = signal_mDOM/signal_DOM


distance = 200
z_ref = 5
coincidences = np.arange(1,9)
PV = [] # p-value
Z = [] # z-score
DH = [] # detection horizon

for coinc in coincidences:
    #plot_distance(coinc)
    PV.append(get_PV(distance, coinc))
    Z.append(get_Z(distance, coinc))
    DH.append(get_distance(coinc, z_ref))

PV, Z, DH = np.array(PV), np.array(Z), np.array(DH)

if debug1:
    fig, ax = plt.subplots(1,1)
    ax2 = ax.twinx()
    ax.plot(coincidences, PV, 'x', color = "C0")
    ax2.plot(coincidences, Z, 'o', color = "C1")
    ax.set_xlabel(r'multiplicity $n_{PMT}$')
    ax.set_ylabel('p-value')
    ax.set_xticks(coincidences)
    ax2.set_ylabel('Z-score')
    plt.title("p-value for different mDOM trigger conditions at {:.0f} kpc distance".format(distance))
    plt.savefig("./plots/significance_{:.0f}kpc.png".format(distance))
    plt.show()

if debug2:
    fig, ax = plt.subplots(1,1)
    ax.plot(coincidences, DH, 'x', color = "C0")
    ax.set_xlabel(r'multiplicity $n_{PMT}$')
    ax.set_ylabel('detection horizon [kpc]')
    ax.set_xticks(coincidences)
    plt.title(r"{:.0f}$\sigma$ detection horizon for different mDOM trigger conditions".format(z_ref))
    plt.savefig("./plots/horizon_{:.0f}sigma.png".format(z_ref))
    plt.show()