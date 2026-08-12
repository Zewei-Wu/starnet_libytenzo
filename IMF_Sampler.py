# this file should be python-3 compatible


import numpy as np
import time

class IMF_sampler():
    """
        This class produces samples from an IMF that matches Enzo's PopIII IMF
        with a number of stars per region that matches the statistical distribution 
        of the Phoenix simulations.  Birth times, masses, and counts per region are pulled
        from CDFs that were generated from the PHX sims--either supplied as arguments or 
        on disk as text files.
    """
    def __init__(self, mchar=20, 
                mmax=300, 
                mmin=1, 
                maxtime=12, 
                dt=6, 
                num_cdf = [], 
                num_bins = None,
                mass_cdf = [],
                mass_bins = None,
                time_cdf = [],
                time_bins = None,
                nstar_mean = 1.311,
                nstar_std = 0.352):
        """

        """
        if num_bins is None:
            num_cdfbins = np.loadtxt('./resources/CDF_Nstar.txt')
            num_cdf = num_cdfbins[0]
            num_bins = num_cdfbins[1]
        if mass_bins is None:
            mass_cdfbins = np.loadtxt('./resources/CDF_mass.txt')
            mass_cdf = mass_cdfbins[0]
            mass_bins = mass_cdfbins[1]
        if time_bins is None:
            time_cdfbins = np.loadtxt('./resources/CDF_creationtime.txt')
            time_cdf = time_cdfbins[0]
            time_bins = time_cdfbins[1]
        self.nstar_mean = nstar_mean
        self.nstar_std = nstar_std
        self.model_time = maxtime
        self.imf_mchar = mchar
        self.imf_mmax = mmax
        self.imf_mmin = mmin
        self.mass_cdf = mass_cdf
        self.mass_cdf_bins = mass_bins
        self.ctime_cdf = time_cdf
        self.ctime_cdf_bins = time_bins
        self.nstar_cdf = num_cdf
        self.nstar_cdf_bins = num_bins
        self.massbins = [1, 11, 20, 40, 100, 140, 200, 260, 300]
        self.timebins = np.arange(0, maxtime, dt)
        self.dt = dt
        self.imf_logfile = "./imf_sampler.log"
        self.test_repeatability = False # flag to initialize with fixed seed for repeatable results
                                        # Otherwise, use time as seed to ensure max randomness
        self.ncalls = 0
    def generate_samples(self, ds, nsamples, birthtime = True):
            self.ncalls += 1
            if self.test_repeatability:
                np.random.seed(int(ds.current_redshift*10000+self.ncalls))
            else:
                np.random.seed(int(time.time()))
            counts = self.get_starcounts(nsamples)
            # print("Counts: ", counts)
            masses = self.get_mstars(counts)
            metal_yield = self.determine_yields(masses)
            times = self.get_birthtimes(counts)
            sample_tokens = self.tokenize(masses, times, birthtime)
            print("IMF generate_samples:")
            print("found counts: %d"%counts[0])
            print("found masses: ", masses)
            print("Metal_yield: %0.3f" % metal_yield[0])
            with open(self.imf_logfile, 'a') as f:
                f.write("%s"%(', '.join(['%d'%t for t in sample_tokens])))
            return sample_tokens, metal_yield[0], sum(masses[0])
            
            
    def get_starcounts(self, nsamples):
        unisamples = np.random.uniform(0,1, size=nsamples)
        indices = np.digitize(unisamples, self.nstar_cdf)
        # indices = np.array([np.argmin(np.abs(self.nstar_cdf - u)) for u in unisamples])
        counts = self.nstar_cdf_bins[indices]
        return counts
        
        
    def get_mstars(self, nstars):
        masses = []
        for nstar in nstars:
            unisamples = np.random.uniform(0,1,size=int(nstar))
            indices = np.digitize(unisamples, self.mass_cdf)
            masses.append(self.mass_cdf_bins[indices].tolist())
        return masses
    
    def metal_yield_sne(self, mass):
        return 0.1077 + 0.3383 * (mass-11)

    def metal_yield_hne(self, mass):
        hneMetals = [3.36, 3.53, 5.48, 7.03, 8.59]
        hneMass = [19.99, 25, 30, 35, 40.01]
        bin = np.searchsorted(hneMass, mass)-1
        # print("mstar = %0.2f Hne metal: bin = %d; len(hneMass) = %d"%(mass, bin, len(hneMass)))
        f = (hneMass[bin]-mass) / (hneMass[bin+1] - hneMass[bin])
        return hneMetals[bin] + f * (hneMetals[bin+1] - hneMetals[bin])

    def metal_yield_pisne(self, mass):
        HeCore = (13./24.) * (mass-20)
        return 5.0+1.304*(HeCore - 64)
    
    def determine_yields(self, masses):
        # sums all the yields
        sum_metal = []
        for smass in masses:
            summ = 0
            for mass in smass:
                # this is a mass-dependent metal yield
                if mass > 11 and mass < 20:
                    summ += self.metal_yield_sne(mass)
                if mass > 20 and mass < 40:
                    summ += self.metal_yield_hne(mass)
                if mass > 140 and mass < 260:
                    summ += self.metal_yield_pisne(mass)
            sum_metal.append(summ)
        return sum_metal
    def get_birthtimes(self, nstars):
        times = []
        for nstar in nstars:
            unisamples = np.random.uniform(0,1,size=int(nstar))
            indices = np.digitize(unisamples, self.ctime_cdf)
            times.append(self.ctime_cdf_bins[indices].tolist())
        return times
    
    def tokenize(self, masses, times, birthtime = True): #times are already in tokens
        # print("MASSES: ", masses)
        # print("TIMES: ", times)
        mass_tok = None
        time_tok = None
        for i in range(len(masses)):
            m_tok,_ = np.histogram(masses[i], bins=self.massbins)
            t_tok,_ = np.histogram(times[i], bins=self.timebins)
            if i == 0:
                mass_tok = m_tok
                time_tok = t_tok
            else:
                mass_tok = np.vstack([mass_tok, m_tok])
                time_tok = np.vstack([time_tok, t_tok])
        if birthtime:
            return np.append(mass_tok, time_tok)
        else:
            return mass_tok
 