from turtle import color
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from sensors import *

def grid_plot(pmt_range, modules_range, detection_horizon, false_alarm_rate):

    loss = np.sqrt(1/detection_horizon**2 + globals.fit_weight_OR * (globals.DES_false_alarm_rate_comb - false_alarm_rate)**2)

    fig, ax = plt.subplots(1,2, figsize = (15,5))

    xx, yy = np.meshgrid(pmt_range, modules_range)
    clp = ax[0].scatter(xx, yy, c = detection_horizon.T, s = 10, marker = 'o', alpha = 0.5, cmap = 'viridis', norm=LogNorm())
    ax[0].set_xlabel(r'$n_{trigger}$')
    ax[0].set_ylabel(r'$N_{\nu}$')
    ax[0].set_xticks(pmt_range)
    clb = plt.colorbar(clp, ax=ax[0])
    clb.set_label('detection horizon [kpc]', fontsize = 10)
    clp = ax[1].scatter(xx, yy, c = 100*false_alarm_rate.T, s = 10, marker = 'o', alpha = 0.5, cmap = 'viridis', norm=LogNorm())
    ax[1].set_xlabel(r'$n_{trigger}$')
    ax[1].set_ylabel(r'$N_{\nu}$')
    ax[1].set_xticks(pmt_range)
    clb = plt.colorbar(clp, ax=ax[1])
    clb.set_label('false alarm rate [1/century]', fontsize = 10)
    plt.tight_layout()


    return fig, ax

def contour_plot(pmt_range, modules_range, detection_horizon, false_alarm_rate):

    fig, ax = plt.subplots(1,2, figsize = (15,5))
    ind = int(len(modules_range)/2)

    for j in range(len(pmt_range)):
        ax[0].plot(modules_range, detection_horizon[j,:], 'o-', ms = 5, alpha = 0.5, label = r"$n_{trigger}$"+" = {:.0f}".format(pmt_range[j]))
        ax[0].set_xlabel('number of modules')
        ax[0].set_ylabel(r'detection horizon $d_{50\%}$')
        ax[0].set_yscale('log')
        ax[0].legend()
        ax[1].plot(modules_range, 100*false_alarm_rate[j,:], 'o-', ms = 5, alpha = 0.5, label = r"$n_{trigger}$"+" = {:.0f}".format(pmt_range[j]))
        ax[1].axhline(1, color = "grey", ls = '--')
        ax[1].text(60, 1, s = "1/century")
        ax[1].set_xlabel('number of modules')
        ax[1].set_ylabel(r'false alarm rate [1/century]')
        ax[1].set_yscale('log')
        ax[1].legend()
        plt.tight_layout()

    return fig, ax

def result_plot(mdom, wls, detector):

    fig, axs = plt.subplots(1,2, figsize = (8,4))
    ax1 = axs[0]
    ax2 = axs[1]
    distance = np.linspace(5, 400, 1000)

    
    Z_WLS = wls.significance(distance)
    P_SN_WLS = wls.detection_probability(distance)

    P_SN_mDOM = np.zeros_like(distance)
    Z_mDOM = np.zeros_like(distance)
    P_SN_comb = np.zeros_like(distance)
    Z_comb = np.zeros_like(distance)
    for i,dd in enumerate(distance):
        P_SN_mDOM[i] = mdom.detection_probability(dd)
        Z_mDOM[i] = mdom.significance(dd)
        P_SN_comb[i] = detector.detection_probability(dd)
        Z_comb[i] = detector.significance(dd)

    ax1.plot(distance, Z_mDOM, label = "mDOM", color = 'red', alpha = 0.5, ls = ':')
    ax1.plot(distance, Z_WLS, label = "WLS", color = 'red', alpha = 0.5, ls = '--')
    ax1.plot(distance, Z_comb, label = "Comb.", color = 'red', ls = '-')
    ax2.plot(distance, P_SN_mDOM, label = "mDOM", color = 'red', alpha = 0.5, ls = ':')
    ax2.plot(distance, P_SN_WLS, label = "WLS", color = 'red', alpha = 0.5, ls = '--')
    ax2.plot(distance, P_SN_comb, label = "Comb.", color = 'red', ls = '-')

    ind5 = np.argmin(np.abs(Z_comb-5))
    ind50 = np.argmin(np.abs(P_SN_comb-0.5))

    ax1.axhline(5, color = 'grey', ls = '--')
    ax1.axvline(distance[ind5], color = 'grey', ls = '--')
    ax1.text(200, 5, s = r"5$\sigma$")

    ax2.axhline(0.5, color = 'grey', ls = '--')
    ax2.axvline(distance[ind50], color = 'grey', ls = '--')
    ax2.text(distance[ind50], 0.5, s = r"50$\%$")

    ax1.set_xlabel("Distance d [kpc]")
    ax1.set_ylabel("Significance Z")
    ax1.set_yscale("log")
    ax2.set_xlabel("Distance d [kpc]")
    ax2.set_ylabel(r"SN detection probability $P_{SN}$")
    ax1.legend()
    ax2.legend()
    plt.tight_layout()

    return fig, axs

def result_plot_single(type, detector):

    fig, axs = plt.subplots(1,2, figsize = (8,4))
    ax1 = axs[0]
    ax2 = axs[1]
    distance = np.linspace(5, 400, 1000)

    P_SN = np.zeros_like(distance)
    Z = np.zeros_like(distance)
    for i,dd in enumerate(distance):
        P_SN[i] = detector.detection_probability(dd)
        Z[i] = detector.significance(dd)

    ax1.plot(distance, Z, label = type, color = 'red', ls = '-')
    ax2.plot(distance, P_SN, label = type, color = 'red', ls = '-')

    ind10 = np.argmin(np.abs(Z-10))
    ind5 = np.argmin(np.abs(Z-5))
    ind3 = np.argmin(np.abs(Z-3))
    ind50 = np.argmin(np.abs(P_SN-0.5))

    ax1.axhline(10, color = 'grey', ls = '--')
    ax1.axhline(5, color = 'grey', ls = ':')
    ax1.axhline(3, color = 'grey', ls = (0, (3, 5, 1, 5, 1, 5)))
    ax1.axvline(distance[ind10], color = 'grey', ls = '--')
    ax1.axvline(distance[ind5], color = 'grey', ls = ':')
    ax1.axvline(distance[ind3], color = 'grey', ls = (0, (3, 5, 1, 5, 1, 5)))
    ax1.text(200, 10, s = r"10$\sigma$")
    ax1.text(200, 5, s = r"5$\sigma$")
    ax1.text(200, 3, s = r"3$\sigma$")

    ax2.axhline(0.5, color = 'grey', ls = '--')
    ax2.axvline(distance[ind50], color = 'grey', ls = '--')
    ax2.text(distance[ind50], 0.5, s = r"50$\%$")

    ax1.set_xlabel("Distance d [kpc]")
    ax1.set_ylabel("Significance Z")
    ax1.set_yscale("log")
    ax2.set_xlabel("Distance d [kpc]")
    ax2.set_ylabel(r"SN detection probability $P_{SN}$")
    ax1.legend()
    ax2.legend()
    plt.tight_layout()

    return fig, axs

