from scipy.special import gamma
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
%matplotlib inline

plt.rcParams['figure.figsize'] = (8,8)
plt.rcParams['figure.dpi']=300
plt.rcParams['savefig.dpi']=300
mpl.rcParams['axes.titlesize'] = 20 #plot tile
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['legend.fontsize'] = 10
#
# ==============================================================================
# Monte Carlo for Cherenkov emission of random distributed neutrino interactions
# ==============================================================================

def crosssection_strumia(enu):
    # cross section in m^2
    me=511e-3    # MeV
    ee=enu-1.293 # MeV
    pe=np.sqrt(ee**2-me**2)
    return 1e-43*pe*ee*enu**(-0.07056+0.02018*np.log(enu)-0.001953*(np.log(enu))**3)/1e4

def interactions(L,d,emean,alpha):
    # get number of interactions per cubic meters from Luminosity, cross section and target density
    # N_interactions =  n_target*L/(4pi d^2)/<E>*<E^2>*9.52x10-48
    # input: 
    # L:     Luminosity in anti-electron neutrinos in ergs
    # d:     distance in kpc
    # emean: mean anti-eleclectron neutrino energy
    # alpha: mean value of shape parameter
    #
    # examples for luminosities and mean energies in anti-electron neutrinos (normal oscillation) :
    # Pagliaroli:  5.0441e+52 erg; <E_nu> = 12.7 MeV
    # Huedepohl:   1.0224e+52 erg; <E_nu> = 11.4 MeV
    # Livermore:   4.8253e+52 erg; <E_nu> = 15.6 MeV
    # Black Hole:  3.8862e+52 erg; <E_nu> = 22.5 MeV
    
    kpcm    = 3.0857e+19         #(kpc in meters)
    ibd     = 9.52e-48           # inverse beta decay cross section in [m^2/E]
    ergMeV  = 624151.            # erg to MeV
    ntarget = 6.164e28           # [1/m^3] for ice at average temperature (effect of air is not included)
    return ntarget*L/(4.*np.pi*d**2)*(2.+alpha)/(1.+alpha)/emean/kpcm**2*crosssection_strumia(emean)*ergMeV

def FrankTamm(Ekinetic,n,lmin,lmax):
    # lmin, lmax   in nm
    # return value in cm
    # beta**2=1-1/gamma**2
    nmtocm=1e7
    beta2=1-(0.511**2/np.power(0.511+Ekinetic,2))
    return 2.*np.pi/137.*(1.-np.divide(1.,(beta2*n**2)))*(1./lmin -1./lmax)*nmtocm

def SNenergyprob(emean,α,x,power):
    k0 = (α+1)**(α+1)/gamma(α+1)/emean
    if   power == 0: k = 1
    elif power == 1: k=          x   /                   emean
    elif power == 2: k= (α+1)*   x**2/((α+2)*            emean**2)
    elif power == 3: k= (α+1)**2*x**3/((α+2)*(α+3)*      emean**3) 
    elif power == 4: k= (α+1)**3*x**4/((α+2)*(α+3)*(α+4)*emean**4)
    else: 
        k=0
        print("power",power,"not implemented")
    return k0*k*(x/emean)**α*np.exp(-(α+1)*x/emean)


def maxSNenergyprob(emean,α,power):
    if   power == 0: k = (α+0)
    elif power == 1: k = (α+1)
    elif power == 2: k = (α+2)
    elif power == 3: k = (α+3)
    elif power == 4: k = (α+4)
    else: 
        k=0
        print("power",power,"not implemented")
    return emean*k/(α+1),SNenergyprob(emean,α,emean*k/(α+1),power)

def randomSNenergy(emean,α,emin,emax,power):
    #calculates random energy for given emean, α and power (=2 for positron)
    loop=True
    while loop:
        e=np.random.uniform(low=emin,high=emax, size=1)[0]
        p=np.random.uniform(low=0.0,high=maxSNenergyprob(emean,α,power)[1], size=1)[0]
        if p < SNenergyprob(emean,α,e,power):
            loop = False
    return e

def fold2(ngamma,eff):
    # provides probability to see 2 fold coincidence for ngamma hitting WOM
    tempx = np.arange(0,20,1)
    tempy = np.array([0,0,0.745,0.947,0.985,0.997,0.999,1,1,1,1,1,1,1,1,1,1,1,1,1])
    return np.interp(ngamma*eff, tempx, tempy, period=None)

def fold3(ngamma,eff):
    # provides probability to see 2 fold coincidence for ngamma hitting WOM
    tempx = np.arange(0,20,1)
    tempy = np.array([0,0,0,0.368,0.671,0.798,0.923,0.95,0.983,\
                      0.984,0.99,0.999,0.997,0.999,0.999,1,1,1,1,1])
    return np.interp(ngamma*eff, tempx, tempy, period=None)

def fold4(ngamma,eff):
    # provides probability to see 2 fold coincidence for ngamma hitting WOM
    tempx = np.arange(0,20,1)
    tempy = np.array([0,0,0,0,0.107,0.229,0.388,0.505,0.636,0.706,\
                      0.788,0.817,0.872,0.917,0.92,0.942,0.96,0.97,0.981,0.987])
    return np.interp(ngamma*eff, tempx, tempy, period=None)

# determine capture efficiencies
# ================= set up =============================================
# ------------ supernova energy spectrum -------------------------------
alpha    = 3        # assumed SN neutrino shape parameter
E        = 15       # assumed mean SN neutrino energy
# ------------ interaction volume --------------------------------------
rIBC     = 1        # interaction radius in x,y plane
zIBC     = 1        # interaction region -z, +z
nWOM     = 20000    # number of WOM-traps
nIBC     = 10000    # number of interactions 
# ---------- Cherenkov angle and its smearing --------------------------
Ch_angle = 42.*np.pi/180. # Cherenkov cone angle
Ch_smear = 5.*np.pi/180.  # some 20 degree smearing of Cherenkov cone
# ---------- WOM geometry -----------------------------------------------
lWOM     = 2        # WOM length in m
rWOM     = 0.128    # WOM radius in m
# ---------- WOM detection efficiency -----------------------------------          
e_trans  = 0.237*132.0/89.62  # correct to 2 cm distance (from thesis M. Thiel)
e_TIR    = 0.41     # capture efficiency
e_abs    = 0.363    # average absorption
QE_mean  = 0.0898   # taking into account QE_PMT, QE_WLS, Cherenkov spectrum 200-669nm
# ----------- coincidence probabilities (0-20 photons) ------------------



# ----------- switch on debug plotting ----------------------------------
debug    = False      
# =======================================================================
#
# ================= preparation =========================================
#
# ----------------- get random SN positron energy -----------------------
#
E_positron_list=[]
for i in range(nIBC):
    E_positron_list.append(randomSNenergy(emean=E,α=alpha,emin=1,emax=50,power=2)) # power=2 for positron
E_positron = np.array(E_positron_list)
#
# ----------------- get positron starting positions ---------------------
#    
phi     = np.random.uniform(0.,2.*np.pi,size=nIBC)         # azimutal angle for position
theta   = np.arccos(np.random.uniform(-1.,1,size=nIBC))    # polar angle for position
r       = np.sqrt(np.random.uniform(0.,1.,size=nIBC))*rIBC # 2d radius (take sqrt!->uniform)
x       = r*np.cos(theta)                                  # start x
y       = r*np.sin(theta)                                  # start y
z       = np.random.uniform(-zIBC,zIBC,size=nIBC)          # start z
#
# ----------------- now get positron directions --------------------------
#
phi_e   = np.random.uniform(0.,2.*np.pi,size=nIBC)         # azimutal angle for direction
theta_e = np.arccos(np.random.uniform(-1.,1,size=nIBC))    # polar angle for direction
#
#
# ------------------ calculate number of Cherenkov photons assuming ------
# disregard Poissonian smearing etc. for now 
#
n_Ch    = np.multiply(FrankTamm(E_positron,1.33,200,669)*0.56,E_positron).astype(int)
# print(n_Ch)
#
# ========================================================================
# 
# ================= start Cherenkov Monte Carlo ==========================
# disregard length of track, and Poissonian smearing
# ---- distribute photons around directions according to Cherenkov angle -
#
nhit = []
for i in range(len(phi_e)):                                # loop through positrons
    theta_Ch = np.random.normal(Ch_angle,Ch_smear,n_Ch[i]) # smear around Cherenkov angle
    phi_Ch   = np.random.uniform(0.,2.*np.pi,n_Ch[i])      # uniform distribution in phi
    #
    # these angle are w.r.t. positron direction 
    # transform into original system, calculate directional cosines:
    #
    temp_x   = (np.cos(phi_e[i])*np.cos(theta_e[i])*np.sin(theta_Ch)*np.cos(phi_Ch))\
        -(np.sin(phi_e[i])*np.sin(theta_Ch)*np.sin(phi_Ch))\
        +(np.cos(phi_e[i])*np.sin(theta_e[i])*np.cos(theta_Ch)) 
        
    temp_y   = ((np.sin(phi_e[i])*np.cos(theta_e[i])*np.sin(theta_Ch)*np.cos(phi_Ch))\
        +(np.cos(phi_e[i])*np.sin(theta_Ch)*np.sin(phi_Ch))\
        +(np.sin(phi_e[i])*np.sin(theta_e[i])*np.cos(theta_Ch)))
        
    temp_z   = -(np.sin(theta_e[i])*np.cos(phi_Ch)*np.sin(theta_Ch))+\
        (np.cos(theta_e[i])*np.cos(theta_Ch)) 
    
    if debug and i== 0:
        cos=temp_x*np.cos(phi_e[i])*np.sin(theta_e[i]) +\
            temp_y*np.sin(phi_e[i])*np.sin(theta_e[i])  + temp_z*np.cos(theta_e[i])
        plt.hist(np.arccos(cos)*57.3)
        plt.show()
    
    temp_tan = np.tan(np.arccos(temp_z))                  # tan theta angle of Cherenkov photon
    temp_cos = np.cos(np.arctan2(temp_y,temp_x))          # cos phi angle of Cherenkov photon
    temp_sin = np.sin(np.arctan2(temp_y,temp_x))          # cos phi angle of Cherenkov photon
    #
    # throw events whose distance to the WOM is too large 
    # both in projection on x,y plane and in z 
    # see: http://mathworld.wolfram.com/Point-LineDistance2-Dimensional.html; there are 2 solutions for crossing point
    #
    temp1       = -(x[i]*temp_cos+y[i]*temp_sin)+((x[i]*temp_cos+y[i]*temp_sin)**2+rWOM**2-x[i]**2-y[i]**2)**0.5
    temp2       = -(x[i]*temp_cos+y[i]*temp_sin)-((x[i]*temp_cos+y[i]*temp_sin)**2+rWOM**2-x[i]**2-y[i]**2)**0.5
    temp_sign1  = np.sign(temp1)
    temp        = np.multiply(np.minimum(np.absolute(temp1),np.absolute(temp2)),temp_sign1)  # not sure if this covers all cases
    condition_r = ((x[i]*temp_cos+y[i]*temp_sin)**2+rWOM**2-x[i]**2-y[i]**2) 
    #
    # select only those photons that hit the WOM:
    #
    s_phi_g   = temp_sin[(condition_r > 0) & (temp >-0.5*lWOM*temp_tan) & (temp < 0.5*lWOM*temp_tan)]
    c_phi_g   = temp_cos[(condition_r > 0) & (temp >-0.5*lWOM*temp_tan) & (temp < 0.5*lWOM*temp_tan)]
    t_theta_g = temp_tan[(condition_r > 0) & (temp >-0.5*lWOM*temp_tan) & (temp < 0.5*lWOM*temp_tan)]
    temp_g    = temp[(condition_r > 0) & (temp >-0.5*lWOM*temp_tan) & (temp < 0.5*lWOM*temp_tan)]
    #
    wom_x = (x[i]+np.multiply(temp_g,c_phi_g))                 # position of photon at WOM (x-coordinate)
    wom_y = (y[i]+np.multiply(temp_g,s_phi_g))                 # position of photon at WOM (x-coordinate)
    wom_z = (z[i]+np.divide(temp_g,t_theta_g))                 # position of photon at WOM (x-coordinate)
    nhit.append(len(wom_x))
#
if debug:
    fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
    circle  = plt.Circle((0, 0), rWOM, color='g', fill=False)
    ax.plot(x,y,"ro")
    ax.add_patch(circle)
    ax.plot(wom_x,wom_y,'.')
    plt.show()

    fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
    rectangle  = plt.Rectangle((-lWOM/2, -rWOM),2*rWOM,lWOM,color='g')
    ax.plot(z,x,"ro")
    ax.plot(wom_z,wom_x,'.')
    plt.show()

    fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
    plt.plot(z,y,"ro")
    plt.plot(wom_z,wom_y,'.')
    plt.show()

# =========================== some histogramming ===============================
binwidth=400
bins=np.arange(0,20000,binwidth)
fig1=plt.figure(num=1,dpi=300,facecolor='w',edgecolor='k')
ax1 = fig1.add_subplot(111)
ax2 = ax1.twiny()
nn,bb,_=ax1.hist(nhit,bins=bins,density=True,histtype="step") # sum entries = 1 
ax1.set_yscale('log')
ax1.set_xlabel(r"#$\gamma$'s hitting WOM")
ax2.set_xlabel(r"probability")
ax2.set_xticks( ax1.get_xticks() )
ax2.set_xbound(ax1.get_xbound())
# ----- get efficiency  to detect photon that hit WOM in PMT
eff=e_trans*e_TIR*e_abs*QE_mean
ax2.set_xticklabels([int(round(x * eff)) for x in ax1.get_xticks()])
ax2.set_xlabel("photons hitting WOM-trap")
fig1.subplots_adjust(top=0.95)
plt.grid()
plt.show()

fig2=plt.figure(num=2,dpi=300,facecolor='w',edgecolor='k')
ax1 = fig2.add_subplot(111)
ax2 = ax1.twiny()
ax1.hist(nhit,bins=bins,density=True,stacked=True,cumulative=True,histtype="step") # sum entries = 1 
ax1.set_xlabel(r"#$\gamma$'s hitting WOM")
ax2.set_xlabel(r"probability")
ax2.set_xticks( ax1.get_xticks() )
ax2.set_xbound(ax1.get_xbound())
# ----- get efficiency  to detect photon that hit WOM in PMT
eff=e_trans*e_TIR*e_abs*QE_mean
ax2.set_xticklabels([int(round(x * eff)) for x in ax1.get_xticks()])
ax2.set_xlabel("photons hitting WOM-trap")
fig2.subplots_adjust(top=0.95)
plt.grid()
plt.show()


tempx=(bb[:-1]+bb[1:])/2.        # get bin means from histogram
tempy=nn*binwidth                # get values from cumulative histogram
plt.plot(tempx,np.cumsum(tempy)-np.cumsum(tempy)[0],drawstyle="steps-post",label=r"fraction $\gamma$ hitting DOM")
plt.plot(tempx,np.cumsum(tempy*fold2(tempx,eff)),drawstyle="steps-post",label="2-fold coincidence")
plt.plot(tempx,np.cumsum(tempy*fold3(tempx,eff)),drawstyle="steps-post",label="3-fold coincidence")
plt.plot(tempx,np.cumsum(tempy*fold4(tempx,eff)),drawstyle="steps-post",label="4-fold coincidence")
plt.xlabel(r"#$\gamma$'s hitting WOM")
plt.ylabel("fraction")
plt.legend(loc="best",fontsize=16)
plt.grid()
plt.show()

#
print("probability to detect at least 1 photon =",round((1-np.cumsum(tempy))[0],3))
print("probability for 2-fold coincidence      =",round(np.cumsum(tempy*fold2(tempx,eff))[-1],3))
print("probability for 3-fold coincidence      =",round(np.cumsum(tempy*fold3(tempx,eff))[-1],3))
print("probability for 4-fold coincidence      =",round(np.cumsum(tempy*fold4(tempx,eff))[-1],3))
print("")
# calculate number of interactions in total time range (assume alpha=3 for now, need to do better)
print("Pagliaroli: ", round(interactions(5.0441e+52,10,12.7,3),3)," interactions/m^3")
print("Huedepohl:  ", round(interactions(1.0224e+52,10,11.4,3),3)," interactions/m^3")
print("Livermore:  ", round(interactions(4.8253e+52,10,15.6,3),3)," interactions/m^3")
print("Black Hole: ", round(interactions(1.0834e+53,10,26.7,3),3)," interactions/m^3")
print("")
# calculate number of detected neutrinos (assuming all interesting events are in simulation region, alpha =3, 20000 WOMs)
factor = np.pi*rIBC*2*zIBC * nWOM
print("Pagliaroli: ", int(round(interactions(5.0441e+52,10,12.7,3)*factor,0))," # neutrino IBD interactions")
print("Huedepohl:  ", int(round(interactions(1.0224e+52,10,11.4,3)*factor,0))," # neutrino IBD interacrions")
print("Livermore:  ", int(round(interactions(4.8253e+52,10,15.6,3)*factor,0))," # neutrino IBD interactions")
print("Black Hole: ", int(round(interactions(1.0834e+53,10,26.7,3)*factor,0))," # neutrino IBD interactions")
print("")
# detected number of neutrinos by WOM-trap (2 fold)
f2 = np.cumsum(tempy*fold2(tempx,eff))[-1]
print("Pagliaroli: ", int(round(interactions(5.0441e+52,10,12.7,3)*factor*f2,0))," # neutrinos 2fold")
print("Huedepohl:  ", int(round(interactions(1.0224e+52,10,11.4,3)*factor*f2,0))," # neutrinos 2fold")
print("Livermore:  ", int(round(interactions(4.8253e+52,10,15.6,3)*factor*f2,0))," # neutrinos 2fold")
print("Black Hole: ", int(round(interactions(1.0834e+53,10,26.7,3)*factor*f2,0))," # neutrinos 2fold")
print("")
f3 = np.cumsum(tempy*fold3(tempx,eff))[-1]
print("Pagliaroli: ", int(round(interactions(5.0441e+52,10,12.7,3)*factor*f3,0))," # neutrinos 3fold")
print("Huedepohl:  ", int(round(interactions(1.0224e+52,10,11.4,3)*factor*f3,0))," # neutrinos 3fold")
print("Livermore:  ", int(round(interactions(4.8253e+52,10,15.6,3)*factor*f3,0))," # neutrinos 3fold")
print("Black Hole: ", int(round(interactions(1.0834e+53,10,26.7,3)*factor*f3,0))," # neutrinos 3fold")
print("")
f4 = np.cumsum(tempy*fold4(tempx,eff))[-1]
print("Pagliaroli: ", int(round(interactions(5.0441e+52,10,12.7,3)*factor*f4,0))," # neutrinos 4fold")
print("Huedepohl:  ", int(round(interactions(1.0224e+52,10,11.4,3)*factor*f4,0))," # neutrinos 4fold")
print("Livermore:  ", int(round(interactions(4.8253e+52,10,15.6,3)*factor*f4,0))," # neutrinos 4fold")
print("Black Hole: ", int(round(interactions(1.0834e+53,10,26.7,3)*factor*f4,0))," # neutrinos 4fold")
