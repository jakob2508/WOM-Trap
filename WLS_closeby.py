rates_LOM   = np.array([1.1e8,3.1e4,2.3e3,8.5e1,5.5,2.94e-1,1.6e-2,4.9e-4])/10000
signals_LOM = np.array([2.0e6,2.5e5,8.0e4,3.0e4,1.89e4,1.00e4,4.8e3,2.03e3])/10000
nWOM=20000
nLOM=10000
B=250
relative_eff=0.7  # relative efficiency of WOM vs LOM
deltat=10
broadening = 1.4
distance=np.arange(1,30,1)
# look at everything in 1 s interval (rate=#events)
plt.title("10s interval")

for i in range(len(rates_LOM)):
    plt.plot(distance,100*relative_uncertainty(signals_LOM[i]*nLOM*100/distance**2,rates_LOM[i]*nWOM*deltat,broadening),label="MDOM coincidence level %i"%(i+1))
plt.plot(distance,100*relative_uncertainty(signals_LOM[0]*nWOM*relative_eff*100/distance**2,B*nWOM*deltat,broadening),'b-',linewidth=3,label="WOM")
plt.legend(loc="best")
plt.xlabel("distance [kpc]")
plt.ylabel("relative uncertainty [%]")
plt.show()

deltat=1
fraction_accretion=0.6
plt.title("1s interval")

for i in range(len(rates_LOM)):
    plt.plot(distance,100*relative_uncertainty(signals_LOM[i]*fraction_accretion*nLOM*100/distance**2,rates_LOM[i]*nWOM*deltat,broadening),label="MDOM coincidence level %i"%(i+1))
plt.plot(distance,100*relative_uncertainty(signals_LOM[0]*fraction_accretion*nWOM*relative_eff*100/distance**2,B*nWOM*deltat,broadening),'b-',linewidth=3,label="WOM")
plt.legend(loc="best")
plt.xlabel("distance [kpc]")
plt.ylabel("relative uncertainty [%]")
plt.show()



