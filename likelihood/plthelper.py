import matplotlib.pyplot as plt
import numpy as np

def plot_LLHR(coinc):
    sig_mDOM = sig_mDOM_gen2_heavy[coinc-1]
    sig_WLS = sig_WLS_gen2_heavy

    sigs_IC86 = np.linspace(sig_IC86*0.9,sig_IC86*1.1, 100)
    sigs_mDOM = np.linspace(sig_mDOM*0.9,sig_mDOM*1.1, 100)
    sigs_WLS = np.linspace(sig_WLS*0.9,sig_WLS*1.1, 100)
    sigs = np.array([sigs_IC86, sigs_mDOM, sigs_WLS])

    chi2_IC86 = chi2(sigs_IC86, "IceCube")
    chi2_mDOM = chi2(sigs_mDOM, "mDOM", coinc = coinc)
    chi2_WLS = chi2(sigs_WLS, "WLS")
    chi2 = np.array([chi2_IC86, chi2_mDOM, chi2_WLS])

    titles = ["IceCube", "IceCube-Gen2 mDOM", "IceCube-Gen2 WLS"]
    xlabels = [r"$S_{IC86}$", r"$S_{mDOM}$", r"$S_{WLS}$"]
    fig, axs = plt.subplots(1,3, figsize = (12,3))
    for i in range(3):
        axs[i].plot(sigs[i], chi2[i])
        axs[i].set_xlabel(xlabels[i])
        axs[i].set_ylabel(r"$\chi^2 = -2 ln (\lambda)$")
        axs[i].set_title(titles[i])

    plt.suptitle("One-dimensional log-likelihood ratios, {:.0f}-fold mDOM coincidence".format(coinc))
    plt.tight_layout()
    plt.savefig("./plots/1D_chi2_coinc="+str(coinc)+".png")
    #plt.show()

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
    #plt.show()

def LLH_scan(distance, coinc):
    max_sig_llh, max_sig = max_LLH(distance, coinc)
    sig_range = np.linspace(max_sig-100, max_sig+100,200)
    likelihood = []
    for s in sig_range:
        likelihood.append(-COMB_LLH(s, distance, coinc))
    plt.plot(sig_range, likelihood)
    plt.axvline(max_sig, color = "C1")
    plt.xlabel("S")
    plt.ylabel("-LLH")
    plt.title("LLH scan: distance = {:.0f} kpc, {:.0f}-fold coincidence".format(distance, coinc))
    plt.tight_layout()
    plt.savefig("plots/LLH_scan_{:.0f}kpc_coinc={:.0f}".format(distance, coinc))
    #plt.show()

def fit_accuracy():
    sig_true = 1E6
    coincidence = np.arange(1,9)
    distance = np.linspace(10, 1000, 100)
    accuracy = np.zeros((len(coincidence),len(distance)))
    for i, coinc in enumerate(coincidence):
        for j, dist in enumerate(distance):
            print(coinc, dist)
            foobar, sig_reco = max_LLH(dist,coinc)
            accuracy[i][j] = (sig_reco - sig_true)/sig_true * 100
        plt.plot(distance, accuracy[i], label="coinc={:.0f}".format(coinc))
    plt.xlabel("distance")
    plt.ylabel(r"accuracy $(S_{reco} - S_{true})/S_{true}$ [%]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/fit_accuracy_nipv.png")
    #plt.show()

def comp_plot(types):
    N = 100000
    fig, ax = plt.subplots(1,3, figsize = (12,3))
    for i in range(3):
        type = types[i]
        if type == "S<B":
            B = 0.9 * N
        if type == "S>B":
            B = 0.1 * N
        elif type == "S~B":
            B = 0.5 * N

        S = np.linspace(1,N+1)
        S2 = np.linspace(1,N+1, 5*N)
        llh_gauss = norm.pdf(N, loc = S2+B, scale = np.sqrt(N))
        llh_poisson_discrete = poisson.pmf(N, S+B)
        llh_poisson_continuous = cont_poisson(N, S2+B)

        ax[i].plot(S, llh_poisson_discrete, '.',color = 'r', label = 'disc. Poisson')
        ax[i].plot(S2, llh_poisson_continuous, 'k:', label = 'cont. Poisson')
        ax[i].plot(S2, llh_gauss, 'k--', alpha = 0.5, label = 'Gauss')

        ax[i].set_xlabel('signal')
        ax[i].set_ylabel('likelihood')
        ax[i].set_title(type)
    ax[0].legend()
    plt.suptitle("N = {:.0f}".format(N))
    plt.tight_layout()
    plt.savefig('./plots/comp_N='+str(N)+'.png')
    plt.show()

def llh_plot(detector_type, x_max):

    x = np.linspace(1, x_max, 1000)
    y = llh(x, detector_type)
    max_ind = y.argmax()

    plt.plot(x,y,'.')
    plt.xlabel("signal")
    plt.ylabel("likelihood")
    plt.axvline(x[max_ind], color = 'C1')
    plt.show()