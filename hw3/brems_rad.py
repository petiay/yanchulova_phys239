# -*- coding: utf-8 -*-
# UCSD Phys 239, HW 3
# 111218
# PYMJ

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import argparse
# from scipy.fftpack import fft
from scipy.signal import periodogram as pd

a0 = 5.29e-9        # Bohr radius [cm]
q = 5.e-10          # franklin (esu) [cm^3/2 g^1/2 s^-1]
Z = 1.              # hydrogen ion
m = 1.e-27          # 9.1e-28 g ~ 1e-27
c = 3.e10           # speed of light, [cm/s]


def calc_brems_rad(n, r0, v0, dt, nb):
    """
    """
    
    # Find the path
    x, y, vx, vy, ax, ay, t = find_path(n, dt, r0, v0)

    # Find the power spectrum
    freq, dWdw = find_pwrspec(ax, ay, dt)
        
    return dWdw, freq, t, x, y, vx, vy, ax, ay

def find_path(n, dt, r0, v0):
    """ Calculates the path of the electron  
    """
    x = np.zeros(n);  x[0] = r0[0]         # x pos
    y = np.zeros(n);  y[0] = r0[1]         # y pos
    vx = np.zeros(n); vx[0] = v0[0]        # x velocity
    vy = np.zeros(n); vy[0] = v0[1]        # y velocity
    ax = np.zeros(n); ay = np.zeros(n)     # x acceleration
    fx = np.zeros(n); fy = np.zeros(n)     # y acceleration
    t = np.zeros(n)                        # time interval

    for i in range(n-1):  # Fx ~ v*t / R^3/2 ;  Fy ~ b / R^3/2

        fx[i] = -Z*q**2. * vx[i]* dt / (y[0]**2. + (vx[i]*dt)**2. )**1.5 
        fy[i] = -Z*q**2. * y[0] / (y[0]**2. + (vy[i]*dt)**2. )**1.5

        ax[i] = fx[i] / m;            ay[i] = fy[i] / m
                
        x[i+1] = x[i] + vx[i]*dt + .5*ax[i]*dt**2.
        y[i+1] = y[i] + vy[i]*dt + .5*ay[i]*dt**2.
        vx[i+1] = vx[i] + ax[i]*dt;   vy[i+1] = vy[i] + ay[i]*dt

        t[i+1] = t[i] + dt

    # make sure there are no zeros
    ax[-1] = ax[-2]; ay[-1] = ay[-2]; t[0]=t[1]
    return x, y, vx, vy, ax, ay, t

def find_pwrspec(ax, ay, dt):
    """Calcultes the power spectrum by taking the time series and sampling 
       frequency of the accelleration (ax and ay) using Welch's method and a
       Lomb-Scargle periodogram."""

    a = np.sqrt(ax**2 + ay**2)
    f, dWdw = pd(a, 1/dt, 'boxcar', scaling='spectrum')
    return f, dWdw

# def find_fft(ax, ay, n):
#     """ Calculates the FT of the acceleration. Not used. """
#     # dv is mostly in y, thus can consider only ay (a_perp), but will do both
#     ax_w = np.fft.fft(ax); ay_w = np.fft.fft(ay)
#     a_w = np.zeros(n)
#     for j in range(n): 
#         a_w[j] = np.sqrt(np.real(ax_w[j])**2. + np.real(ay_w[j])**2.)
#     return a_w
# 
# def find_spec(a_w, dt, r0, v0, ax, ay, t):
#     """ Calculates the power spectrum of radiation. Not used."""
#     # dWdw = 2 e^2 a(ω)^2 / (3 c^3 pi)
#     dWdw = 2 * q**2 * a_w**2 / (3 * np.pi * c**3)
#     return dWdw
 

def find_peak(omega, xlim, dWdw):
    """ Finds the frequency at the peak of the power spectrum"""                

    # Limit the range of frequencies                            
    omega_lim_inds = np.where((omega >= xlim[0]) & (omega <= xlim[1]))[0]
    omega_lim = omega[omega_lim_inds]
    dW_lim = dWdw[omega_lim_inds] # find the power at these frequencies
    omega_max = omega_lim[np.where(dW_lim == max(dW_lim))][0] # find omega_max

    return omega_max


def plot_brems_rad(nb=None, v0=None, n=None, x=None, y=None, vx=None, vy=None, 
                   ax=None, ay=None, t=None, dWdw=None, omega=None, 
                   b=None, v=None, omega_b=None, omega_v=None, newfig=True, 
                   plotpath=False, plotspectrum=False, plotvariation=False, 
                   xlim_w=(1.5e14, 8e14), 
                   ms=2, Z=1, pos=411, varystr=None, savefig=False, panel=True):
    """
    nb:         Number of Bohr radii of initial position
    Plots the path of the electron, the spectrum of emitted radiation, and the
    change in peak frequency with impact parameter, b, and initial velocity, v0.
    """

    if plotpath:
        if newfig: 
            fig, ax0 = plt.subplots(6, 2, figsize=(10,10))
            plt.suptitle(r'Q.3 Bremsstrahlung radiation: Electron path, '
                            'velocity, and acceleration', y=0.95)

        x = x/float(a0); y = y/float(a0) # back to Bohr radii, for plotting

        axis = plt.subplot(3, 2, 1)     # position
        axis.plot(x, y, '.', ms=ms, label='$x_{0}=%da_{0}$ \n$y_{0}=%da_{0}$' % (nb[0], nb[1]))
        axis.set_xlabel('x [$a_{0}$]'); axis.set_ylabel('y [$a_{0}$]')
        # axis.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        xlim=axis.get_xlim(); ylim = axis.get_ylim() #plot text
        xpos = xlim[0]+(xlim[1] - xlim[0])*0.04; dy = ylim[1] - ylim[0]; fs=9
        axis.text(xpos, ylim[1]-dy*0.65, '$\\tau$ = %ss' % '{0:1.1e}'.format(tau), fontsize=fs)
        axis.text(xpos, ylim[1]-dy*0.77, 'dt = %s' % '{0:1.1e}'.format(dt), fontsize=fs)
        axis.text(xpos, ylim[1]-dy*0.89, 'n = %s' % n, fontsize=fs)
        axis.legend(numpoints=1,loc=1, handletextpad=0.1, fontsize='small')

        axis = plt.subplot(3, 2, 3)     # velocity
        axis.plot(vx, vy, '-', lw=1)
        axis.plot(vx, vy, '.', ms=ms, label='$v_{x0}=%s$ \n$v_{y0}=%d$' % ('{:.1e}'.format(v0[0]), v0[1]))
        axis.ticklabel_format(style='sci', axis='both')#, scilimits=(0,0))
        axis.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        axis.set_xlabel('$v_{x}$ $[cm/s]$'); axis.set_ylabel('$v_{y}$ $[cm/s]$')
        axis.invert_xaxis()
        axis.legend(numpoints=1, loc=1, handletextpad=0.1, fontsize='small')

        axis = plt.subplot(3, 2, 5)     # acceleration
        axis.plot(ax, ay, '-', lw=1, zorder=8)
        axis.plot(ax, ay, '.', ms=ms, zorder=10)
        axis.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        axis.set_xlabel('$a_{x}$ $[cm/s^{2}]$'); axis.set_ylabel('$a_{y}$ $[cm/s^{2}]$')    

        # Plotting the time dependence
        axx = plt.subplot(6, 2, 2)
        axx.plot(t, x, '.', ms=ms, zorder=10)
        axx.set_ylabel('x [$a_{0}$]')
        axx.get_xaxis().set_visible(False)
        axx.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        axx.yaxis.set_label_position("right")

        axy = plt.subplot(6, 2, 4)
        axy.plot(t, y, '.', ms=ms, zorder=10)
        axy.get_xaxis().set_visible(False)
        axy.set_ylabel('y [$a_{0}$]')
        axy.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        axy.yaxis.set_label_position("right")

        axvx = plt.subplot(6, 2, 6)
        axvx.plot(t, vx, '.', ms=ms, zorder=10)
        axvx.get_xaxis().set_visible(False)
        axvx.set_ylabel('$v_{x}$ $[cm/s]$')        
        axvx.yaxis.set_label_position("right")

        axvy = plt.subplot(6, 2, 8)
        axvy.plot(t, vy, '.', ms=ms, zorder=10)
        axvy.get_xaxis().set_visible(False)
        axvy.set_ylabel('$v_{y}$ $[cm/s]$')
        axvy.yaxis.set_label_position("right")

        axax = plt.subplot(6, 2, 10)
        axax.plot(t, ax, '.', ms=ms, zorder=10)
        axax.get_xaxis().set_visible(False)
        axax.set_ylabel('$a_{x}$ $[cm/s^{2}]$')
        axax.yaxis.set_label_position("right")

        axay = plt.subplot(6, 2, 12)
        axay.plot(t, ay, '.', ms=ms, zorder=10)
        axay.set_xlabel('t [s]'); axay.set_ylabel('$a_{y}$ $[cm/s^{2}]$')
        axay.yaxis.set_label_position("right")

        plt.subplots_adjust(hspace=0.4)
        
        if savefig:
            filename = '1_path.png'
            plt.savefig(filename); plt.close()


    if plotspectrum:
        # scale the spectrum for plotting
        dWdw = np.sqrt(dWdw/max(dWdw))
        dWdw = -dWdw+max(dWdw)

        if panel is False: 
            fig2, ax2 = plt.subplots(1, 1, figsize=(10,10)); pos=111
            grid = plt.GridSpec(1, 1)
        else: # plot panels with varying b or v0 
            if newfig:
                fig2, ax2 = plt.subplots(5, 1, figsize=(10,10))
                plt.suptitle(r'Q.4/5 Bremsstrahlung radiation power spectrum: '
                            'Varying %s' % varystr, y=0.95)

        axis = plt.subplot(pos)
        axis.plot(omega, dWdw)
        axis.set_ylabel('$dW/d\omega/dW/d\omega_{max}$')
        axis.set_xscale("log", nonposx='clip')
        axis.set_xlim(xlim_w)
        axis.invert_xaxis() # short freqencies on the right
        
        if pos == 515 or panel is False: axis.set_xlabel('$\omega[s^{-1}]$')
        else: axis.xaxis.label.set_visible(False)

        # Plot w_cutoff, the "interaction frequency"
        if v and b:
            w_cut = v/b
            if panel:
                xlim=axis.get_xlim(); ylim = axis.get_ylim()
                xpos = xlim[0]+(np.abs(xlim[1] - xlim[0]))*0.0001
                axis.text(xpos, ylim[1]-(ylim[1] - ylim[0])*0.3, 
                         '$x_{0}=%da_{0}$, $y_{0}=%da_{0}$' % (nb[0],nb[1]), fontsize=9)
                axis.text(xpos, ylim[1]-(ylim[1] - ylim[0])*0.5, 
                         '$v_{x0}=%s$, $v_{y0}=%d$' % ('{:.1e}'.format(v0[0]), v0[1]), fontsize=9)
                leglabel = '$\omega_{intxn}=V_{0}/b$=%sHz' % '{:.2e}'.format(w_cut)
            else: leglabel = '$\omega_{intxn}=V_{0}/b=%sHz$ \n'\
                             '$b=%sa_{0}$, $v=%scm/s$' % ('{:.2e}'.format(w_cut), \
                              nb[1], '{:.1e}'.format(v))
            axis.axvline(w_cut, -1, 1, c='r', ls='--', lw=0.5, label=leglabel)
        
        axis.legend(numpoints=1, loc=1, fontsize='small')
        plt.subplots_adjust(hspace=0.4)

        if savefig:
            if panel:
                if varystr == 'impact parameter': filename = '3_spectrum_vary_b.png'
                else: filename = '4_spectrum_vary_v0.png'
            else: filename = '2_spectrum.png'
            plt.savefig(filename); plt.close()


    if plotvariation:
        # scale the frequency for plotting
        omega_b = omega_b/max(omega_b)
        omega_v = omega_v/max(omega_v)
        if newfig: 
            fig3, ax3 = plt.subplots(1, 2, figsize=(10,10))
            plt.suptitle(r'Q.5 Bremsstrahlung radiation: $\omega_{MAX}$ vs impact '
                        'parameter (b) and initial velocity ($V_{0}$)', y=0.95)

        axis = plt.subplot(121) #vary b
        axis.plot(b, omega_b, 'o')
        axis.set_xlabel('$b$ $[a_{0}]$')
        axis.set_ylabel('$\omega_{max}/MAX(\omega_{max})$')
        axis.set_xlim(b[0]-50, b[-1]+50)
        axis.set_ylim(min(omega_b)-0.1, max(omega_b)+0.1)

        axis = plt.subplot(122) #vary v
        axis.plot(v, omega_v, 'o')
        axis.set_xlabel('$V_{0}$ $[cm/s]$')
        axis.set_ylabel('$\omega_{max}/MAX(\omega_{max})$')
        axis.set_xlim(v[0]-5e6, v[-1]+5e6)
        axis.set_ylim(min(omega_v)-0.1, max(omega_v)+0.1)
        
        if savefig:
            filename = '5_vary_b_v.png'
            plt.savefig(filename); plt.close()


if __name__ == '__main__':
    """
    Code can be run in a number of ways:
        1. 'python brems_rad'  ==> produces 4 plots with default parameters and 
            saves them in code directory.
        2. Using options. Example:
            brems_rad --n 10000 --nb 500 500 --v0 3e7 0
            ==> Uses 10000 timesteps, places the e- at x, y = (500, 500) Bohr 
                radii, and gives it an initial velocity Vx, Vy = (3e7, 0) cm/s.
    """
    print ('\n=============================================================')
    print ('\nPhysics 239, HW 3. Bremsstrahlung Radiation')

    parser = argparse.ArgumentParser()

    # Option to save the figure
    ftypes = ['png', 'jpg', 'jpeg', 'pdf']
    parser.add_argument('-s', '--savefig', action='store',
                        default=False, choices=ftypes,
                        help='Save figure with one of the following formats: \
                        "{}"'.format('", "'.join(ftypes)) )
    # Inputs for r0, v0, n
    parser.add_argument('--nb', nargs='+', action='store', default=[500, 500],
                        type=int, help='Enter initial position in Bohr radii\
                        (x(0), y(0))' )
    parser.add_argument('--v0', nargs='+', action='store', default=[1.e7, 0.],
                        type=float, help='Enter initial velocity [cm/s] \
                        (vx(0), vy(0))' )
    parser.add_argument('--n', action='store', default=1000,
                        type=int, help='Enter number of steps (int)' )
    parser.add_argument('--xlim', nargs='+', action='store', default=[1e11, 5e15],
                        type=float, help='Enter limits of frequency axis \
                        --xlim min max' )
    args = parser.parse_args()

    if not args.savefig:
        print ('\nFigures automatically saved in code directory \n\n')
        args.savefig=True

    nb=args.nb; v0=args.v0; n=args.n; xlim=args.xlim
    savefig=args.savefig


# ==================== Set up and perform calculations =========================

    r0 = (nb[0]*a0, nb[1]*a0)                 # initial position
                # Timescale of intxn (collision time): 
                # tau = n*dt ~ 2b/v  ==> dt ~2b/(v*n)
    dt = 2*r0[1] / (v0[0]*n)                  # time interval
    tau = n*dt                                # collision time
    
    # Calculating the Bremsstrahlung radiation
    dWdw, omega, t, x, y, vx, vy, ax, ay = calc_brems_rad(n, r0, v0, dt, nb)


# =================== Plot path, velocity, and acceleration ====================

    print ('... Plotting the path ...\n')
    plot_brems_rad(nb, v0, n, x, y, vx, vy, ax, ay, t, savefig=savefig, plotpath=True)
        
    print ('===> Electron path plots saved to \'1_path.png\' \n\n')

    print ('... Plotting the spectrum ... \n')
    plot_brems_rad(nb, v0, dWdw=dWdw, omega=omega, b=r0[1], v=v0[0],
                   savefig=savefig, plotspectrum=True, panel=False, xlim_w=xlim)

    print ('===> Power spectrum plot saved to \'2_spectrum.png\' \n')    
    
    
# ==== Plot power spectrum: vary impact parameter b and initial velocity v0 ====

    b = np.linspace(300, 600, 5); l_b = len(b)
    v = np.linspace(1e7, 3e7, 5); l_v = len(v)
    

    print ('\n... Plotting the power spectrum, varying b ...\n')
    omega_max_b=[]
    for j in range(l_b):
        newfig=False; savefig=False
        if j==0: newfig=True
        if j==l_b-1: savefig=True
        pos = int('%s1%s' % (l_b, j+1)) # panel plot position
        
        # find dt for this combination of v0 and b
        r0 = (nb[0]*a0, b[j]*a0); dt = 2*r0[1] / (v0[0]*n) 
        dWdw, omega, t, x, y, vx, vy, ax, ay = calc_brems_rad(n, r0, v0, dt, nb)
        plot_brems_rad((nb[0], b[j]), v0, dWdw=dWdw, omega=omega, newfig=newfig, 
                        plotspectrum=True, pos=pos,  b=r0[1], v=v0[0],
                        xlim_w=(1e12,1e14), varystr='impact parameter',
                        savefig=savefig, panel=True)
        omega_max = find_peak(omega, xlim, dWdw) # get peak freq
        omega_max_b.append(omega_max)
        
    print ('===> Radiation power spectrum varying impact parameter plots '
            'saved to \'3_spec_vary_b.png\'')


    print ('\n\n... Plotting the power spectrum, varying v0 ...\n')
    omega_max_v=[]
    for j in range(l_v):
        newfig=False; savefig=False
        if j==0: newfig=True
        if j==l_v-1: savefig=True
        pos = int('%s1%s' % (l_v, j+1))
        
        # find dt for this combination of v0 and b
        v0 = (v[j], v0[1]); dt = 2*r0[1] / (v0[0]*n) 
        dWdw, omega, t, x, y, vx, vy, ax, ay = calc_brems_rad(n, r0, v0, dt, nb)
        plot_brems_rad(nb, v0, dWdw=dWdw, omega=omega, newfig=newfig, 
                        plotspectrum=True, pos=pos,  b=r0[1], v=v0[0],
                        xlim_w=(1e12,1e14), varystr='initial velocity',
                        savefig=savefig, panel=True)
        omega_max = find_peak(omega, xlim, dWdw)
        omega_max_v.append(omega_max)

    print ('===> Radiation power spectrum varying initial velocity plots '
            'saved to \'4_spec_vary_v.png\' ')

# ============ Plot how peak of power spectrum varies with b and v =============

    plot_brems_rad(b=b, v=v, omega_b=omega_max_b, omega_v=omega_max_v, 
                    plotvariation=True, savefig=savefig)

    print ('\n===> Variation in peak frequency with varying b and v0 '
            'saved to \'5_vary_b_v.png\' ')

    plt.close('all') #close all plots
    
    print ('\nEnd of code.\n')