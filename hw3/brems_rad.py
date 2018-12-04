# -*- coding: utf-8 -*-
# UCSD Phys 239, HW 3
# 111218
# PYMJ

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import argparse
import sys

a0 = 5.29e-9        # Bohr radius [cm]
q = 5.e-10          # franklin (esu) [cm^3/2 g^1/2 s^-1]
Z = 1.              # hydrogen ion
m = 1.e-27          # 9.1e-28 g ~ 1e-27
c = 3.e10           # speed of light, [cm/s]


def calc_brems_rad(n, r0, v0, dt, nb):
    """ Calculates the path and spectrum for Bremsstrahlung radiation."""
    
    # Find the path
    x, y, vx, vy, ax, ay, t, fx, fy = find_path(n, dt, r0, v0)

    # Find the Fourier transform of the acceleration
    a_w = find_fft(ax, ay, dt)

    # Find the power spectrum
    dWdw = find_spec(a_w, dt, r0, v0, ax, ay, t)

    freq = 1/t
    
    return dWdw, freq, t, x, y, vx, vy, ax, ay, fx, fy

def find_path(n, dt, r0, v0):
    """ Calculates the path of the electron. """
    x = np.zeros(n);  x[0] = r0[0]         # x pos
    y = np.zeros(n);  y[0] = r0[1]         # y pos
    vx = np.zeros(n); vx[0] = v0[0]        # x velocity
    vy = np.zeros(n); vy[0] = v0[1]        # y velocity
    ax = np.zeros(n); ay = np.zeros(n)     # x acceleration
    fx = np.zeros(n); fy = np.zeros(n)     # y acceleration
    t = np.zeros(n);                       # time interval

    for i in range(n-1):
        fx[i] = -Z*q**2. * x[i] / (y[0]**2. + x[i]**2. )**1.5 
        fy[i] = -Z*q**2. * y[0] / (y[0]**2. + x[i]**2. )**1.5

        ax[i] = fx[i] / m;            ay[i] = fy[i] / m
                
        x[i+1] = x[i] + vx[i]*dt + .5*ax[i]*dt**2.
        y[i+1] = y[i] + vy[i]*dt + .5*ay[i]*dt**2.
        vx[i+1] = vx[i] + ax[i]*dt;   vy[i+1] = vy[i] + ay[i]*dt

        t[i+1] = t[i] + dt

    # make sure there are no zeros
    ax[-1] = ax[-2]; ay[-1] = ay[-2]; t[0]=t[1]; fx[-1]= fx[-2]; fy[-1]=fy[-2]
    return x, y, vx, vy, ax, ay, t, fx, fy

def find_fft(ax, ay, n):
    
    a = np.sqrt(ax**2 + ay**2)
    a_w = np.real(np.fft.fft(a))
    return a_w


def find_spec(a_w, dt, r0, v0, ax, ay, t):
    """ Calculates the power spectrum of radiation. Not used."""
    # dWdw = 2 e^2 a(ω)^2 / (3 c^3 pi)

    dWdw = 2 * q**2 * a_w**2 / (3 * np.pi * c**3)    
    return dWdw
 

def find_freq_peak(omega, xlim, dWdw):
    """ Finds the frequency at the peak of the power spectrum"""                

    # Limit the range of frequencies       
    if xlim is None: xlim = (omega[0], omega[1])     
    omega_lim_inds = np.where((omega >= xlim[0]) & (omega <= xlim[1]))[0]
    omega_lim = omega[omega_lim_inds]
    dW_lim = dWdw[np.where((omega >= xlim[0]) & (omega <= xlim[1]))[0]]
    # check if the peak of the spectrum is within default freq range
    if len(dW_lim)==0:
        print ('\n\n  *** Default frequency limits are outside of the max '\
                'spectrum frequency range *** \n\n '\
                'Try changing the default frequency range with \'--xlim <xmin> <xmax>\'' \
                '\n Path is saved to 1_path.png. Exiting.')
        sys.exit()
        
    dWdw_max = max(dW_lim)
    omega_max = omega_lim[np.where(dW_lim == dWdw_max)][0] # find omega_max
    return omega_max, dWdw_max


def plot_brems_rad(nb=None, v0=None, n=None, x=None, y=None, vx=None, vy=None, 
                   ax=None, ay=None, fx=None, fy=None, t=None, dWdw=None, omega=None, 
                   b=None, v=None, omega_b=None, omega_v=None, newfig=True, 
                   plotpath=False, plotspectrum=False, plotvariation=False, 
                   xlim_w=None, ylim_w=None, figure=None, omega_max=None,
                   ms=2, Z=1, pos=411, varystr=None, savefig=False, panel=True,
                   varyparam=None, varyaxes=[0.67, 0.42, 0.22, 0.22]):
    """
    Plots the path of the electron, the force along the path, the spectrum of 
    the emitted radiation, and the change in peak frequency with impact 
    parameter, b, and initial velocity, v0.
    """

    if plotpath:
        if newfig: 
            fig, ax0 = plt.subplots(6, 3, figsize=(11,10))
            plt.suptitle(r'Q.3 Bremsstrahlung radiation: Electron path, '
                            'velocity, acceleration, and force', y=0.95)

        x = x/float(a0); y = y/float(a0) # back to Bohr radii, for plotting

        # set x and y axes limits
        if x[0]<0: xlo = x[0]+0.2*x[0]; xhi = np.abs(xlo)#sigx=-1
        else: xlo = -(x[0]+0.2*x[0]); xhi = x[0]+0.2*x[0]
        if y[0]<0: ylo = y[0]+0.2*y[0]; yhi = np.abs(ylo)
        else: ylo = -(y[0]+0.2*y[0]); yhi = y[0]+0.2*y[0]
        lim_x = (xlo,xhi)
        lim_y = (ylo,yhi)

        # Plot position, velocity and acceleration in x and y
        axis = plt.subplot(3, 3, 1)             # position
        axis.plot(0, 0, '+', c='orange', ms=18) # ion is at 0,0
        axis.plot(x[0], y[0], 'go', ms=5)       # starting position
        axis.plot(x[-1], y[-1], 'ro', ms=5)     # ending position
        axis.plot(x, y, '.', ms=ms, label='$x_{0}=%da_{0}$ \n$y_{0}=%da_{0}$' % (nb[0], nb[1]))
        axis.set_xlabel('x [$a_{0}$]'); axis.set_ylabel('y [$a_{0}$]')
        
        axis.set_xlim(lim_x) # write out calculation parameters: dt, n, and tau
        axis.set_ylim(lim_y)
        xlim=axis.get_xlim(); ylim = axis.get_ylim()
        xpos = xlim[0]+(xlim[1] - xlim[0])*0.04; dy = ylim[1] - ylim[0]; fs=8
        axis.text(xpos, ylim[1]-dy*0.70, '$\\tau=2b/v_{0}=$%ss' % '{0:1.1e}'.format(tau), fontsize=fs)
        axis.text(xpos, ylim[1]-dy*0.82, 'dt = %s' % '{0:1.1e}'.format(dt), fontsize=fs)
        axis.text(xpos, ylim[1]-dy*0.94, 'n = %s' % n, fontsize=fs)
        axis.legend(numpoints=1, loc=1, handletextpad=0.1, fontsize='small', frameon=False)

        axis = plt.subplot(3, 3, 4)     # velocity
        axis.plot(vx, vy, '.', ms=ms, label='$v_{x0}=%s$ \n$v_{y0}=%d$' % ('{:.1e}'.format(v0[0]), v0[1]))
        axis.plot(vx[0], vy[0], 'go', ms=5) 
        axis.plot(vx[-1], vy[-1], 'ro', ms=5) 
        axis.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        axis.set_xlabel('$v_{x}$ $[cm/s]$'); axis.set_ylabel('$v_{y}$ $[cm/s]$')
        axis.legend(numpoints=1, loc=6, handletextpad=0.1, fontsize='small')

        axis = plt.subplot(3, 3, 7)     # acceleration
        axis.plot(ax, ay, '.', ms=ms, zorder=10)
        axis.plot(ax[0], ay[0], 'go', ms=5) 
        axis.plot(ax[-1], ay[-1], 'ro', ms=5) 
        axis.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        axis.set_xlabel('$a_{x}$ $[cm/s^{2}]$'); axis.set_ylabel('$a_{y}$ $[cm/s^{2}]$')    


        # Plotting the time dependence
        axx = plt.subplot(6, 3, 2)     #x     
        axx.plot(t, x, '.', ms=ms, zorder=10)
        axx.get_xaxis().set_visible(False)
        axx.set_ylabel('x [$a_{0}$]')
        axis.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        axy = plt.subplot(6, 3, 5)      #y
        axy.plot(t, y, '.', ms=ms, zorder=10)
        axy.get_xaxis().set_visible(False)
        axy.set_ylabel('y [$a_{0}$]')
        axis.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        axvx = plt.subplot(6, 3, 8)     #vx
        axvx.plot(t, vx, '.', ms=ms, zorder=10)
        axvx.get_xaxis().set_visible(False)
        axvx.set_ylabel('$v_{x}$ $[cm/s]$')   

        axvy = plt.subplot(6, 3, 11)    #vy
        axvy.plot(t, vy, '.', ms=ms, zorder=10)
        axvy.get_xaxis().set_visible(False)
        axvy.set_ylabel('$v_{y}$ $[cm/s]$')

        axax = plt.subplot(6, 3, 14)    #ax
        axax.plot(t, ax, '.', ms=ms, zorder=10)
        axax.get_xaxis().set_visible(False)
        axax.set_ylabel('$a_{x}$ $[cm/s^{2}]$')

        axay = plt.subplot(6, 3, 17)    # ay
        axay.plot(t, ay, '.', ms=ms, zorder=10)
        axay.set_xlabel('t [s]'); axay.set_ylabel('$a_{y}$ $[cm/s^{2}]$')
        
        axfx = plt.subplot(2, 3, 3)     # Fx
        axfx.set_title('Force in x')
        axfx.plot(x, fx, '.', ms=ms, zorder=10)
        axfx.set_ylabel('Fx'); axfx.set_xlabel('x')
        axfx.set_xlim(-2000,10000)
        
        spec = plt.subplot(2, 3, 6)     # Power spectrum
        spec.set_title('Power Spectrum')
        # normalize to max dWdw in the freq range of interest
        omega_max, dWdw_max = find_freq_peak(omega, xlim_w, dWdw)
        dWdw = dWdw/dWdw_max
        
        spec.plot(omega, dWdw)
        spec.set_ylabel('$dW/d\omega/dW/d\omega _{max}$')
        spec.set_xlabel('$\omega[s^{-1}]$')
        if xlim_w is not None: spec.set_xlim(xlim_w)
        spec.set_ylim(-0.1, 1.1)

        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        
        if savefig:
            filename = '1_path_spectrum.png'
            plt.savefig(filename); plt.close()


    if plotspectrum:
        # dWdw = dWdw/max(dWdw) # Normalize the spectrum for plotting

        if newfig:
            fig2 = plt.figure(figsize=(8,8))
            plt.suptitle(r'Q.4/5 Bremsstrahlung radiation power spectrum: '
                        'Varying %s' % varystr, y=0.95)

        leglabel = '$b=%sa_{0}$, $v=%scm/s$' % (nb[1], '{:.1e}'.format(v0[0]))

        plt.plot(omega, dWdw, label=leglabel)
        plt.ylabel('$dW/d\omega/dW/d\omega _{max}$')
        if xlim_w is not None: plt.xlim(xlim_w)
        if ylim_w is not None: plt.ylim(ylim_w)
        plt.ylim(-0.1, 1.1)
        plt.xlabel('$\omega[s^{-1}]$')
        plt.legend(numpoints=1, loc=1, fontsize='small')

        if newfig: return fig2


    if plotvariation:
        l, b, w, h = varyaxes # adding an inset axis
        axvary = figure.add_axes([l, b, w, h])

        if varystr == 'impact parameter': 
            xlabel = '$b$ $[a_{0}]$'; filename = '2_spectrum_vary_b.png'
        else: xlabel='$V_{0}$ $[cm/s]$'; filename = '3_spectrum_vary_v0.png'
                    
        axvary.plot(varyparam, omega_max, 'o')
        axvary.set_xlabel(xlabel); axvary.set_ylabel('$\omega/\omega_{max}$')

        if savefig:
            plt.savefig(filename); #plt.close()


if __name__ == '__main__':
    """
    Code can be run in a number of ways:
        1. 'python brems_rad.py'  ==> produces 3 plots with default parameters 
            and saves them in code directory.
        2. Using options. Example:
            brems_rad.py --n 10000 --nb -500 500 --v0 3e7 0
            ==> Uses 10000 timesteps, places the e- at x, y = (-500, 500) Bohr 
                radii, and gives it an initial velocity Vx, Vy = (3e7, 0) cm/s.
                
        *** Note that plots 2 and 3 works well with the default values since to 
            pick out the frequency at the max power spectrum the frequency range
            has been limited specific to these input parameters.
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
    parser.add_argument('--nb', nargs='+', action='store', default=[-2000, 500],
                        type=int, help='Enter initial position in Bohr radii\
                        (x(0), y(0))' )
    parser.add_argument('--v0', nargs='+', action='store', default=[2.4e7, 0.],
                        type=float, help='Enter initial velocity [cm/s] \
                        (vx(0), vy(0))' )
    parser.add_argument('--n', action='store', default=12000,
                        type=int, help='Enter number of steps (int)' )
    parser.add_argument('--xlim', nargs='+', action='store', default=[8.3381e10, 8.348e10],
                        type=float, help='Enter limits of frequency axis \
                        --xlim min max' )
    args = parser.parse_args()
    #ylim = [0, 2e-8]

    if not args.savefig:
        print ('\nFigures will save in code directory \n')
        args.savefig=True

    nb=args.nb; v0=args.v0; n=args.n; xlim=args.xlim
    savefig=args.savefig


# ==================== Set up and perform calculations =========================

    r0 = (nb[0]*a0, nb[1]*a0)                        # initial position

                # Timescale of intxn (collision time): 
                # tau = n*dt ~ 2b/v  ==> dt ~2b/(v*n)
    tau = np.abs(2*r0[1]) / (v0[0])                  # collision intxn time
    dt = 1e-15                                       # time step
    
    # Calculating the Bremsstrahlung radiation
    dWdw, omega, t, x, y, vx, vy, ax, ay, fx, fy = calc_brems_rad(n, r0, v0, dt, nb)

    # trim dWdw and omega to discard most zero values of the spectrum
    # dWdw = dWdw[-int(0.01*n):-1]; omega = omega[-int(0.01*n):-1]
    
    
# =================== Plot path, velocity, and acceleration ====================

    print ('... Plotting the path ...')
    plot_brems_rad(nb, v0, n, x, y, vx, vy, ax, ay, fx, fy, t, dWdw, omega, 
                    savefig=savefig, xlim_w=xlim, plotpath=True)

    print ('===> Electron path plots saved to \'1_path_spectrum.png\' \n')
    
    
# ==== Plot power spectrum: vary impact parameter b and initial velocity v0 ====

    b = np.linspace(300, 1500, 5); l_b = len(b)
    v = np.linspace(1.7e7, 2.6e7, 5); l_v = len(v)


    print ('... Plotting the power spectrum, varying b ...')
    omega_max_b=[]; varystr='impact parameter'
    for j in range(l_b):
        newfig=False; savefig=False
        if j==0: newfig=True
        if j==l_b-1: savefig=True
        xlim=(8.33850e10, 8.348e10)
        r0 = (nb[0]*a0, b[j]*a0)
        dWdw, omega, t, x, y, vx, vy, ax, ay, fx, fy = calc_brems_rad(n, r0, v0, dt, nb)

        omega_max, dWdw_max = find_freq_peak(omega, xlim, dWdw) # get peak freq
        omega_max_b.append(omega_max)
        dWdw = dWdw/dWdw_max # normalize spectrum

        fig_b = plot_brems_rad((nb[0], b[j]), v0, dWdw=dWdw, omega=omega, 
                        newfig=newfig, plotspectrum=True, omega_b=omega_max, 
                        xlim_w=xlim, varystr=varystr, savefig=savefig,
                        ylim_w=(-0.03, 0.41))
        if j==0: figvary=fig_b

# ============ Plot how peak of power spectrum varies with b ===================
    for j in range(l_b):
        omega_max = omega_max_b[j]/max(omega_max_b)
        plot_brems_rad(b=b[j], v=v, omega_max=omega_max, plotvariation=True, 
                   savefig=savefig, figure=figvary, varyparam=b[j], varystr=varystr)    

    print ('===> Radiation power spectrum varying impact parameter plot saved to '
            ' \'2_spec_vary_b.png\'')


    xlim_w = [4.5e13, 1.67e14]
    print ('\n... Plotting the power spectrum, varying v0 ...')
    omega_max_v=[]; varystr='initial velocity'
    for j in range(l_v):
        newfig=False; savefig=False
        if j==0: newfig=True
        if j==l_v-1: savefig=True
        xlim=(8.3370e10, 8.348e10)
        v0 = (v[j], v0[1])
        dWdw, omega, t, x, y, vx, vy, ax, ay, fx, fy = calc_brems_rad(n, r0, v0, dt, nb)
        
        omega_max, dWdw_max = find_freq_peak(omega, xlim, dWdw) # get peak freq
        omega_max_v.append(omega_max)
        dWdw = dWdw/dWdw_max # normalize spectrum
        
        fig_v = plot_brems_rad(nb, v0, dWdw=dWdw, omega=omega, newfig=newfig, 
                        plotspectrum=True, xlim_w=xlim, panel=False, 
                        omega_v=omega_max, ylim_w=(-0.03, 0.14), varystr=varystr)
        if j==0: figvary=fig_v
        
# ============ Plot how peak of power spectrum varies with v0 ==================
    for j in range(l_v):
        omega_max = omega_max_v[j]/max(omega_max_v)
        plot_brems_rad(b=b, v=v[j], omega_max=omega_max, plotvariation=True, 
                    savefig=savefig, figure=figvary, varyparam=v[j], 
                    varystr=varystr)#,varyaxes=[0.45, 0.18, 0.17, 0.17])

    print ('===> Radiation power spectrum varying initial velocity plots '
            'saved to \'3_spec_vary_v.png\' ')

    plt.close('all') #close all plots
    
    print ('\nEnd of code.\n')