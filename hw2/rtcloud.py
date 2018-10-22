# UCSD Phys 239, HW 2
# 101918
# PYMJ

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import sys
import argparse

d=100*3.0857e18    # Cloud depth [cm]
n_density=1        # n = 1 [cm-3]
N = n_density * d  # N = n * d [cm-2]

# sig_nu = 3.24e-21  # 

# sig_tau_thin = sig_nu_calc(tau_nu=1e-3)
# sig_tau_thick = sig_nu_calc(tau_nu=1e3)
# sig_tau = sig_nu_calc()


def sig_nu_calc(tau_nu=1):
    """ Calculates the absorption cross-section of a cloud of 
        column density N and total optical depth tau.
    tau_nu = n * sig_nu * D = N * sig_nu
    => sig_nu = tau_nu / N
    """
    sig_nu = tau_nu / N
    return sig_nu

def sig_nu_fn(sig_nu_0=3.24e-21, n_steps=1000, fwhmdiv=10, nu_mean=None, 
              nu_min=1e14, nu_max=1e16, stdev=None, fwhm=None, newfig=False, 
              plot=False, label='', ms=2, plotlog=False, 
              savefig=False, pos=311, color=None):
    """ Generates a cross section sig_nu as a fn of frequency, nu, and passes
    this on to find_I_nu to find and plot the specific intensity as a fn of
    sig_nu, I_nu_0, and S_nu.

    """
    
    nu = np.linspace(nu_min, nu_max, n_steps)
    if fwhm is None: fwhm = (nu_max - nu_min) / fwhmdiv
    if stdev is None: stdev = fwhm / 2.355
    if nu_mean is None: nu_mean = np.mean(nu)
    
    # Creating sig_nu Gaussian
    sig_nu = sig_nu_0 * 1./(stdev * np.sqrt(2.*np.pi)) * np.exp(-(nu-nu_mean)**2./(2.*stdev**2.))

    # Scaling the sig_nu dependence on nu to satisfy the condition that the
    # maximum of the Gaussian is equal to the input sig_nu_0
    sig_nu_max = max(sig_nu)
    scalefactor = sig_nu_0 / sig_nu_max
    sig_nu = sig_nu_0 * scalefactor * 1./(stdev * np.sqrt(2.*np.pi)) * np.exp(-(nu-nu_mean)**2./(2.*stdev**2.))

    if plot:
        if newfig is True: 
            fig, ax0 = plt.subplots(3, 1, sharex=True, figsize=(9,9))
            plt.suptitle(r'Q.3 Cross section $\sigma_{\nu}$', y=0.99)
        ax = plt.subplot(pos)
        ax.plot(nu, sig_nu, '.', label=label, ms=ms, c=color)    
        if plotlog: 
            ax=plt.gca()
            ax.set_xscale("log", nonposx='clip'); 
        if label: plt.legend()
        plt.show()
        
        plt.ylabel(r'$\sigma_{\nu}$ $[cm^{-2}]$', fontsize=12)
        if pos==313: 
            plt.xlabel(r'${\nu}$ $[Hz]$', fontsize=12)
            plt.tight_layout()
        if savefig:
            filename = '3_sig_nu_plots.png'
            plt.savefig(filename)
            plt.close()
            print ('\n\n3. ===> Cross section plots saved in \'%s\' \n' % filename)
    return sig_nu, nu, nu_mean


def find_I_nu(sig_nu_0=None, n_steps=1000, I_nu_0=2, S_nu=1, n=None, const=False,
              tau_nu_0=1, fwhmdiv=10, nu_0=None, nu_min=1e14, nu_max=1e16,
              plot=True, newfig=False, plotlog=False, savefig=False,
              plot_I_nu_0=True, plot_tau_nu=False, plot_nu=True, ms=2, pos=321):
    """
    Reads in values of 
    1. sig_nu (cross-section for absorption)
    2. I_nu_0 (the background specific intensity at s=0, where the cloud begins)
    3. S_nu (the source function)
    and calculates the specific intensity at s = D, I_nu(D) be using n_steps
    intermediate values.
    
    sig_nu_0:   float
        Cross section; default=None
    I_nu_0:     float
        Initial (background) intensity; default=2
    S_nu:       float
        The source function; default=1
    n_steps:    int
        Number of steps, where the step size, ds = dist / n_steps
    n:          int
        Number density; default 1 atom/cm^3
    const:      boolean
        Flag to calculate the specific intensity at one frequency by taking as
        input sig_nu, I(0), and S; default=False
    tau_nu_0:   int
        Optical depth at frequency nu_0; default=1
    fwhmdiv:    int
        Full-width at half-max divisor to ensure that nu_2-nu_1 >> delta_nu(FWHM)
    nu_0:       float
        Central frequency of the Gaussian line profile
        
    Variables to ensure various plotting options follow.
    """

    if sig_nu_0 is None: sig_nu_0 = sig_nu_calc(tau_nu=tau_nu_0)
    if n is None: n=n_density
    
    dist = np.linspace(0, d, n_steps) # distance through the cloud, [cm]
    I_nu = np.empty((n_steps,))       # initial intensity
    
    # Obtain the cross section's dependence on frequency
    sig_nu, nu, nu_mean = sig_nu_fn(sig_nu_0, n_steps, nu_mean=nu_0,
                                    nu_min=nu_min, nu_max=nu_max)

    I_nu[0] = I_nu_0 # 1st element in I_nu is the initial intensity, I_nu(0)
    
    if const:
        sig_nu = sig_nu_0 # pin cross section at a single value at nu_0: sig_nu_0
        for i in range(n_steps-1):
            I_nu[i+1] = I_nu_0 * np.exp(-n*sig_nu*d) + S_nu * (1 - np.exp(-n*sig_nu*d))
        plot=False
    else:
        for i in range(n_steps-1): # Solving the transfer equation
            I_nu[i+1] = I_nu_0 * np.exp(-n * sig_nu[i] * d) + \
                     S_nu * (1 - np.exp(-n * sig_nu[i] * d) )

    # Plotting instructions
    if plot:
        if newfig: 
            fig, ax0 = plt.subplots(3, 2, sharex=True, figsize=(10,11))
            plt.suptitle(r'Q.4 Specific intensity $I_{\nu}(D)$', y=0.97)
        ax = plt.subplot(pos)
        ax.plot(nu, I_nu, '.', ms=ms, zorder=10)

        if pos == 323:
            plt.ylabel(r'$I_{\nu}$ $[ergs$ $s^{-1}$ $cm^{-2}$ $ster^{-1}$ $Hz^{-1}]$', fontsize=12)
        if pos == 325 or pos == 326:
            plt.xlabel(r'${\nu}$ $[Hz]$', fontsize=12)
        if pos == 321: 
            tau_label = r'$\tau_{\nu}(D)$ >> 1'; plot_tau_nu=True; plot_nu = False
        else: tau_label = r'$\tau_{\nu_{0}}=%s$' % tau_nu_0

        plt.axhline(S_nu, xmax=max(nu), c='r', ls=':', label=r'$S_{\nu}$')        
        if plot_I_nu_0: plt.axhline(I_nu_0, xmax=max(nu), c='b', ls=':', label=r'$I_{\nu}(0)$')        

        if plot_tau_nu: plt.plot(nu[0], I_nu[0], c='c', ls='', label=tau_label)
        if plot_nu: plt.axvline(nu_mean, ls=':', lw=1, c='gray')

        plt.ylim(-0.1, 2.2)
        plt.legend(fontsize=8, loc=4)
        if savefig:
            filename = '4_I_nu_plots.png'
            plt.savefig(filename); plt.close()
            print ('\n4. ===> Specific intensity plots saved in \'%s\' \n' % filename)

    return I_nu      
        

if __name__ == '__main__':
    
    print ('\n\n=============================================================')
    print ('\nPhysics 239, HW 2')
    parser = argparse.ArgumentParser()

    # Option to save the figure
    ftypes = ['png', 'jpg', 'jpeg', 'pdf']
    parser.add_argument('-s', '--savefig', action='store',
                        default=False, choices=ftypes,
                        help='Save figure with one of the following formats: \
                        "{}"'.format('", "'.join(ftypes)) )

    # Optional inputs for Q 2: finding I_nu(D) at one frequency
    parser.add_argument('--sig_nu', action='store', default=3.24e-21,
                        type=float, help='Choose a value for the initial intensity \
                        (float)' )
    parser.add_argument('--I_nu_0', action='store', default=2,
                        type=float, help='Choose a value for the initial intensity \
                        (Any int or float)' )
    parser.add_argument('--S_nu', action='store', default=1,
                        type=float, help='Choose a value for the source function \
                        (Any int or float)' )
    args = parser.parse_args()

    if not args.savefig:
        print ('Figures will be automatically saved in code directory')
        args.savefig=True


# ========== Q.1 - Calculating the column density and cross section ============
    sig_nu_thin = sig_nu_calc(tau_nu=1e-3)
    sig_nu_thick = sig_nu_calc(tau_nu=1e3)
    sig_nu_1 = sig_nu_calc(tau_nu=1)

    print ('\n1. The column density of the cloud is N = n * D = %s cm^-2' % N)
    
    print (' The cross section is sig = N * tau:\
            \n For tau_nu = 10e-3, sig_nu %s cm^2 \
            \n For tau_nu = 10e3 : %s cm^2 \
            \n For tau_nu = 1 : %s cm^2 \n' % 
            ('{0:.3g}'.format(sig_nu_thin), \
            '{0:.3g}'.format(sig_nu_thick), \
            '{0:.3g}'.format(sig_nu_1)) )

    
# ================= Q.2 - Finding I_nu(D) at one frequency, nu =================
    n_steps = 1000      # number of steps = # elements in array
    nu_mean = 1.e15     # Central frequency of the Gaussian line profile
    
    sig_nu_in = args.sig_nu     # cross section
    I_nu_0 = args.I_nu_0        # initial intensity
    S_nu = args.S_nu            # source fn
    
    # Call to find_I_nu() with 'const'=True for finding I_nu at a specific freq.
    I_nu = find_I_nu(sig_nu_in, I_nu_0=I_nu_0, S_nu=S_nu, const=True)

    print ('\n2. The specific intensity at D = 100pc, sig_nu = %s, I(0) = %s, and S = %s is: ' % ('{0:.3g}'.format(sig_nu_in), I_nu_0, S_nu))
    print ('    I(D) = %s ergs s^-1 cm^-2 ster^-1 Hz^-1  \n' % '{0:.3g}'.format(I_nu[-1]))
    
    print ('The default input values were:')
    print ('  - cross section, sig_nu = %s' % sig_nu_in)
    print ('  - initial intensity, I_nu_0 = %s' % I_nu_0)
    print ('  - source function, S_nu = %s \n' % S_nu)
    
    print ('To input other values run code as:')
    print ('$python rtcloud.py --sig_nu <sig_nu_val> --I_nu_0 <I_nu_0_val> --S_nu <S_nu_val>')

    
    
# ===== Q.3 - Generate plots of sig_nu as a fn of nu using values from 1. ======

    nu_min = 1e13; nu_max=5e14; fwhmd=30
    
    sig_nu = sig_nu_fn(sig_nu_thin, n_steps, nu_min=nu_min, nu_max=nu_max, fwhmdiv=fwhmd,
                        plot=True, pos=311, newfig=True, label=r'$\tau_{\nu}=10^{-3}$')
    sig_nu = sig_nu_fn(sig_nu_1, n_steps, nu_min=nu_min, nu_max=nu_max, fwhmdiv=fwhmd,
                        plot=True, pos=312, label=r'$\tau_{\nu}=1$', color='orange')
    sig_nu = sig_nu_fn(sig_nu_thick, n_steps, nu_min=nu_min, nu_max=nu_max, fwhmdiv=fwhmd,
                        plot=True, pos=313, label=r'$\tau_{\nu}=10^{3}$', color='m', 
                        savefig=args.savefig)
        
# ===== Q.4 - Generating different cloud scenarios with resulting I_nu(D) =====
    
    # Choose initial parameters and ranges; tau_nu(D) < 1 for all except a)
    
    # a) tau_nu >> 1    =>   n >> 1
    n=1e35
    find_I_nu(newfig=True, n=n, S_nu=2, plot_I_nu_0=False)
    
    # b) I_nu_0 = 0
    find_I_nu(I_nu_0 = 0, S_nu=2, pos=322)
    
    # c) I_nu_0 < S_nu
    find_I_nu(I_nu_0 = 1, S_nu=2, pos=323)
    
    # d) I_nu_0 > S_nu
    find_I_nu(I_nu_0 = 2, S_nu=1, pos=324)
    
    # e) I_nu_0 < S_nu; tau_nu_0 > 1
    find_I_nu(I_nu_0 = 1, S_nu=2, tau_nu_0=10, plot_tau_nu=True, pos=325)

    # f) I_nu_0 > S_nu; tau_nu_0 > 1
    find_I_nu(I_nu_0 = 2, S_nu=1, tau_nu_0=100, plot_tau_nu=True, pos=326,
                savefig=args.savefig)
                
    print ('\nEnd of code.\n')