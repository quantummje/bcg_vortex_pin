#
# Newton-Rahpson-Richardson iterator
# 2D homogeneous vortex
#

import numpy as np
#import scipy as sp
from scipy import sparse
from scipy.sparse import bmat
from scipy.sparse.linalg import bicgstab
import matplotlib.pyplot as plt
from numpy import linalg as la
from dyn_h import rltm
from anim import anim
import time

#
def set_sp(Dx,Dy):
    #
    Nx = int(2*Dx[0]/Dx[1] + 1)
    Ny = int(2*Dy[0]/Dy[1] + 1)
    #
    (dx, dy) = (Dx[1], Dy[1])
    #
    # harmonic trap
    x = np.linspace(-Dx[0],Dx[0],Nx)
    y = np.linspace(-Dy[0],Dy[0],Ny)

    xx, yy = np.meshgrid(x,y,indexing='ij')

    # partial derivative matrices
    d2x = sparse.diags([1,-2,1],[-1,0,1],shape=(Nx,Nx),format='csr')

    d2x[0,0] = -1.
    d2x[Nx-1,Nx-1] = -1.
    d2x = d2x / dx**2
    #
    d2y = sparse.diags([1,-2,1],[-1,0,1],shape=(Ny,Ny),format='csr')

    d2y[0,0] = -1.
    d2y[Ny-1,Ny-1] = -1.
    d2y = d2y / dy**2
    #
    # identity matrices
    Ix = sparse.diags([1],[0],shape=(Nx,Nx),format='csr')
    Iy = sparse.diags([1],[0],shape=(Ny,Ny),format='csr')
    #
    # 2D discrete laplacian
    d2 = sparse.kron(d2x,Iy) + sparse.kron(Ix,d2y)
    #
    return d2
#

#
def get_heaviside(x1,x2):
    #
    if x1 < 0.0:
        #
        out = 0.0
    #
    elif x1 == 0.0:
        #
        out = np.float(x2)
    #
    elif x1 > 0.0:
        #
        out = 1.0
    #
    return out
#

#
def build_pot(Dx,Dy,R0):
    #
    Nx = int(2*Dx[0]/Dx[1] + 1)
    Ny = int(2*Dy[0]/Dy[1] + 1)
    #
    x = np.linspace(-Dx[0],Dx[0],Nx)
    y = np.linspace(-Dy[0],Dy[0],Ny)
    #
    xx, yy = np.meshgrid(x,y)
    #
    r = pow(pow(xx,2) + pow(yy,2),.5)
    #
    heaviside = np.vectorize(get_heaviside)
    #
    hvs_pr = 0.5*(1.0 + np.tanh((r + R0)/0.25)) # heaviside(r + R0,1.0)
    hvs_mr = 0.5*(1.0 + np.tanh((r - R0)/0.25)) # heaviside(r - R0,1.0)
    #
    pot = 1.0 - (hvs_pr - hvs_mr)
    #
    return pot.T, x, y
#

#
def get_ini(Dx,Dy,r1,r2,g,N,R0,typ):
    #
    Nx = int(2*Dx[0]/Dx[1] + 1)
    Ny = int(2*Dy[0]/Dy[1] + 1)
    #
    (dx, dy) = (Dx[1], Dy[1])
    #
    x = np.linspace(-Dx[0],Dx[0],Nx)
    y = np.linspace(-Dy[0],Dy[0],Ny)

    xx, yy = np.meshgrid(x,y,indexing='ij')

    rsq_1 = pow(pow(xx-r1[0],2) + pow(yy-r1[1],2),.5)
    rsq_2 = pow(pow(xx-r2[0],2) + pow(yy-r2[1],2),.5)

    #
    psi_out = np.zeros((Nx,Ny),dtype='complex')
    #
    pot, _, _, = build_pot(Dx=Dx,Dy=Dy,R0=R0)
    #
    if typ == 0:
        #
        psi = np.tanh(2.5*rsq_1) + np.tanh(2.5*rsq_2)
        #
        phi = np.arctan2(yy-r1[1],xx-r1[0]) - np.arctan2(yy-r2[1],xx-r2[0])
        #
        psi_out = (1.0 - pot) * pow(psi,2) * np.exp(1j*phi)
        #
        psi_out = psi_out * pow(dx*dy*np.trapz(np.trapz(np.abs(psi_out)**2)),-.5)
    #
    elif typ == 1:
        #
        psi_out = -pow(np.pi,-.5)*np.exp(-0.5*(pow(xx,2) + pow(yy,2))) * (xx + 1j*yy)
        psi_out = psi_out * pow(dx*dy*np.trapz(np.trapz(np.abs(psi_out)**2)),-.5)
    #

    return psi_out
#

#
def get_mu(Dx,Dy,g,R0,V0,psi_in):
    #
    Nx = int(2*Dx[0]/Dx[1] + 1)
    Ny = int(2*Dy[0]/Dy[1] + 1)
    #
    (dx, dy) = (Dx[1], Dy[1])
    #
    x = np.linspace(-Dx[0],Dx[0],Nx)
    y = np.linspace(-Dy[0],Dy[0],Ny)
    #
    ke = 0.
    pe = 0.
    en = 0.
    vdw = 0.
    #
    xx, yy = np.meshgrid(x,y,indexing='ij')
    #
    pot, _, _y, = build_pot(Dx=Dx,Dy=Dy,R0=R0)
    V = V0 * pot
    #
    for jj in range(1,Nx-1):
        #
        for kk in range(1,Ny-1):
            #
            kx = np.abs((psi_in[jj+1,kk] - psi_in[jj-1,kk])/(2*dx))**2
            ky = np.abs((psi_in[jj,kk+1] - psi_in[jj,kk-1])/(2*dy))**2
            #
            ke = ke + 0.5 * dx*dy*(kx + ky)
            #
            pe = pe + dx*dy*V[jj,kk]*np.abs(psi_in[jj,kk])**2
            #
            vdw = vdw + g*dx*dy*np.abs(psi_in[jj,kk])**4
            #
        #
    #
    nm = dx*dy*np.trapz(np.trapz(np.abs(psi_in)**2))
    mu = (ke + pe + vdw)/nm
    #
    return mu
#

#
def get_ind(Dx,Dy,r0):
    #
    Nx = int(2*Dx[0]/Dx[1] + 1)
    Ny = int(2*Dy[0]/Dy[1] + 1)
    #
    x = np.linspace(-Dx[0],Dx[0],Nx)
    y = np.linspace(-Dy[0],Dy[0],Ny)
    #
    if r0[0] > 0.:
        #
        ind_x = np.where((x > 0.99*r0[0]) & (x < 1.01*r0[0]))[0][0]
    #
    elif r0[0] < 0.:
        #
        ind_x = np.where((x < 0.99*r0[0]) & (x > 1.01*r0[0]))[0][0]
    #
    elif r0[0] == 0.:
        #
        ind_x = int((Nx-1)/2)
    #
    if r0[1] > 0.:
        #
        ind_y = np.where((y > 0.99*r0[1]) & (y < 1.01*r0[1]))[0][0]
    #
    elif r0[1] < 0.:
        #
        ind_y = np.where((y < 0.99*r0[1]) & (y > 1.01*r0[1]))[0][0]
    #
    elif r0[1] == 0.:
        #
        ind_y = int((Ny-1)/2)
    #
    return ind_x, ind_y
#

#
def get_res(Dx,Dy,r1,r2,g,mu,R0,V0,psi_in,pflag):
    #

    Nx = int(2*Dx[0]/Dx[1] + 1)
    Ny = int(2*Dy[0]/Dy[1] + 1)
    #

    x = np.linspace(-Dx[0],Dx[0],Nx)
    y = np.linspace(-Dy[0],Dy[0],Ny)
    #

    d2 = set_sp(Dx=Dx,Dy=Dy)
    r = np.zeros((2*Nx*Ny,1),dtype='float')
    #

    pot, _, _, = build_pot(Dx=Dx,Dy=Dy,R0=R0)

    k_rr = -0.5*d2*np.real(psi_in).ravel()
    k_ii = -0.5*d2*np.imag(psi_in).ravel()
    #

    p_rr = ((V0*pot + g*(pow(np.real(psi_in),2) + pow(np.imag(psi_in),2)) - mu)*np.real(psi_in)).ravel()
    p_ii = ((V0*pot + g*(pow(np.real(psi_in),2) + pow(np.imag(psi_in),2)) - mu)*np.imag(psi_in)).ravel()
    #

    r[:Nx*Ny,0] = k_rr + p_rr
    r[Nx*Ny:,0] = k_ii + p_ii

    if pflag == True:
        #
        r_tmp = r[:Nx*Ny,0].reshape(Nx,Ny)
        i_tmp = r[Nx*Ny:,0].reshape(Nx,Ny)

        #
        ind_x1, ind_y1 = get_ind(Dx=Dx,Dy=Dy,r0=r1)
        ind_x2, ind_y2 = get_ind(Dx=Dx,Dy=Dy,r0=r2)

        r_tmp[ind_x1,ind_y1] = 0.
        i_tmp[ind_x1,ind_y1] = 0.

        r_tmp[ind_x2,ind_y2] = 0.
        i_tmp[ind_x2,ind_y2] = 0.

        r[:Nx*Ny,0] = r_tmp.reshape(Nx*Ny)
        r[Nx*Ny:,0] = i_tmp.reshape(Nx*Ny)
    #

    return r
#

#
def get_jac(Dx,Dy,r1,r2,g,mu,R0,V0,psi_in,pflag):
    #

    Nx = int(2*Dx[0]/Dx[1] + 1)
    Ny = int(2*Dy[0]/Dy[1] + 1)
    NN = int(Nx*Ny)
    #
    x = np.linspace(-Dx[0],Dx[0],Nx)
    y = np.linspace(-Dy[0],Dy[0],Ny)
    #

    dl2 = set_sp(Dx=Dx,Dy=Dy)

    pot, _, _, = build_pot(Dx=Dx,Dy=Dy,R0=R0)
    #

    k_dd = -0.5*dl2
    #

    p_rr = sparse.diags((V0*pot + g*(3*pow(np.real(psi_in),2) + pow(np.imag(psi_in),2)) - mu).ravel(),0,format='csr')
    p_ii = sparse.diags((V0*pot + g*(pow(np.real(psi_in),2) + 3*pow(np.imag(psi_in),2)) - mu).ravel(),0,format='csr')
    #

    p_ri = sparse.diags((2*g*np.real(psi_in)*np.imag(psi_in)).ravel(),0,format='csr')
    p_ir = sparse.diags((2*g*np.imag(psi_in)*np.real(psi_in)).ravel(),0,format='csr')
    #

    j = bmat([[k_dd+p_rr,p_ri],[p_ir,k_dd+p_ii]],format='csr')

    j0 = k_dd + p_rr
    j1 = k_dd + p_ii

    #

    if pflag == True:
        #
        ind_y1, ind_x1 = get_ind(Dx=Dx,Dy=Dy,r0=r1)
        ind_y2, ind_x2 = get_ind(Dx=Dx,Dy=Dy,r0=r2)

        nv = int(2)

        ind = np.zeros(nv,dtype=int)

        rw1 = np.zeros(nv,dtype=int)
        rw2 = np.zeros(nv,dtype=int)

        i1 = np.zeros(nv,dtype=int)
        i2 = np.zeros(nv,dtype=int)

        d1 = np.zeros(nv,dtype=int)
        d2 = np.zeros(nv,dtype=int)

        v1 = np.zeros(nv,dtype=float)
        v2 = np.zeros(nv,dtype=float)

        s1 = np.zeros(nv,dtype=int)
        s2 = np.zeros(nv,dtype=int)
        s3 = np.zeros(nv,dtype=int)
        s4 = np.zeros(nv,dtype=int)

        tmp_1 = int(ind_x1 + ind_y1*Ny)
        tmp_2 = int(ind_x2 + ind_y2*Ny)
        #

        if tmp_1 > tmp_2:
            #
            ind[0] = tmp_2
            ind[1] = tmp_1
        #
        elif tmp_1 < tmp_2:
            #
            ind[0] = tmp_1
            ind[1] = tmp_2
        #

        for kk in range(0,ind.shape[0]):
            #
            rw1[kk] = j.getrow(ind[kk]).data.shape[0]

            t1_lis = j.getrow(ind[kk]).data.tolist()

            if np.any(t1_lis) == True:
                #
                t1_val = [i for i in t1_lis if t1_lis.count(i) == 1][0]

                i1[kk] = np.where(j.getrow(ind[kk]).data == t1_val)[0][0]

                if np.where(j.data == t1_val)[0].shape[0] != 0 and kk != 1:
                    #
                    d1[kk] = np.where(j.data == t1_val)[0][kk]
                #
                elif np.where(j.data == t1_val)[0].shape[0] != 0 and kk == 1:
                    #
                    d1[kk] = np.where(j.data == t1_val)[0][1]
                #
                else:
                    d1[kk] = np.where(j.data == t1_val)[0][0]
                #

                v1[kk] = j.data[d1[kk]]

                s1[kk] = int(d1[kk] - i1[kk])
                s2[kk] = int(s1[kk] + rw1[kk])
                #
                for jj in range(s1[kk],s2[kk],1):
                    #
                    if np.any(t1_lis) == True:
                        #
                        j.data[jj] = 0.
                    #
                #
            #
            rw2[kk] = j.getrow(ind[kk]+NN).data.shape[0]

            t2_lis = j.getrow(ind[kk]+NN).data.tolist()

            if np.any(t2_lis) == True:
                #
                t2_val = [i for i in t2_lis if t2_lis.count(i) == 1][0]

                i2[kk] = np.where(j.getrow(ind[kk]+NN).data == t2_val)[0][0]

                if np.where(j.data == t2_val)[0].shape[0] != 0 and kk != 1:
                    #
                    d2[kk] = np.where(j.data == t2_val)[0][kk]
                #
                elif np.where(j.data == t2_val)[0].shape[0] != 0 and kk == 1:
                    #
                    d2[kk] = np.where(j.data == t2_val)[0][1]
                #
                else:
                    d2[kk] = np.where(j.data == t2_val)[0][0]
                #

                v2[kk] = j.data[d2[kk]]
                #

                s3[kk] = int(d2[kk] - i2[kk])
                s4[kk] = int(s3[kk] + rw2[kk])
                #
            #
        #
        for kk in range(0,ind.shape[0]):
            #
            for jj in range(s3[kk],s4[kk],1):
                #
                if np.any(t2_lis) == True:
                    #
                    j.data[jj] = 0.
                #
            #
            if np.any(t1_lis) == True:
                #
                j.data[d1[kk]] = v1[kk]
            #
            elif np.any(t2_lis) == True:
                #
                j.data[d2[kk]] = v2[kk]
            #
        #
    #
    return j, j0, j1, p_ri, p_ir, k_dd
#

#
def do_pri(Dx,Dy,r1,r2,g,N,R0,V0,tol,cpar,bcg_tol,tau,typ,dflag,pflag):
    #
    Nx = int(2*Dx[0]/Dx[1] + 1)
    Ny = int(2*Dy[0]/Dy[1] + 1)

    #
    (dx, dy) = (Dx[1], Dy[1])

    #
    x = np.linspace(-Dx[0],Dx[0],Nx)
    y = np.linspace(-Dy[0],Dy[0],Ny)

    #
    psi0 = get_ini(Dx=Dx,Dy=Dy,r1=r1,r2=r2,g=g,N=N,R0=R0,typ=typ)
    mu0 = get_mu(Dx=Dx,Dy=Dy,g=g,R0=R0,V0=V0,psi_in=psi0)

    psi = psi0
    mu = mu0

    #
    ind_x1, ind_y1 = get_ind(Dx=Dx,Dy=Dy,r0=r1)
    ind_x2, ind_y2 = get_ind(Dx=Dx,Dy=Dy,r0=r2)

    psi[ind_x1,ind_y1] = 0.
    psi[ind_x2,ind_y2] = 0.

    #N0 = dx*dy*np.trapz(np.trapz(np.abs(psi0)**2))
    #n0 = N0 / (2*Dx[0]*2*Dy[0])

    #
    print('\psi initial norm: ' + repr(dx*dy*np.trapz(np.trapz(np.abs(psi0)**2))))
    #

    print('Numerical mu: ' + repr(mu0))
    print('Analytical mu: ' + repr(mu))

    print('Vort #1 \psi(' + repr(r1[0]) + ',' + repr(r1[1]) + ') = ' + repr(psi[ind_x1,ind_y1]))
    print('Vort #2 \psi(' + repr(r2[0]) + ',' + repr(r2[1]) + ') = ' + repr(psi[ind_x2,ind_y2]))

    #print('Initial \psi(0,Ny) = ' + repr(psi[0,int(Ny-1)]))
    #
    run_flag = True
    c = 0
    #
    fn_dat = []
    psix_dat = []
    psiy_dat = []
    #
    cmin = cpar[0]
    cmax = cpar[1]
    csmp = cpar[2]
    #
    while run_flag:
        #
        j_in, j0, j1, p_ri, p_ir, k_dd = get_jac(Dx=Dx,Dy=Dy,r1=r1,r2=r2,g=g,mu=mu,R0=R0,V0=V0,psi_in=psi,pflag=pflag)
        r_in = get_res(Dx=Dx,Dy=Dy,r1=r1,r2=r2,g=g,mu=mu,R0=R0,V0=V0,psi_in=psi,pflag=pflag)

        #
        t0 = time.time()
        dpsi, ec = bicgstab(j_in, -tau*r_in, tol=bcg_tol)
        print('bcgstab time: ' + repr(time.time() - t0))
        #
        psi = psi + dpsi[:Nx*Ny].reshape(Nx,Ny) + 1j*dpsi[Nx*Ny:].reshape(Nx,Ny)

        #print('chemical potential: ' + repr(get_mu(Dx=Dx,Dy=Dy,g=g,psi_in=psi)))
        #
        fn_dat = np.append(fn_dat,la.norm(dpsi,2))
        #
        c = c + 1
        #
        if divmod(c,csmp)[1] == 0:
            #
            print('------------------------')
            print('Iteration no: ' + repr(c))
            print('Frob. norm: ' + repr(la.norm(dpsi,2)))
            print('\psi norm: ' + repr(dx*dy*np.trapz(np.trapz(np.abs(psi)**2))))
            print('max(\psi): ' + repr(np.max(abs(psi))))
            print('Vort #1 \psi(' + repr(r1[0]) + ',' + repr(r1[1]) + ') = ' + repr(psi[ind_x1,ind_y1]))
            print('Vort #2 \psi(' + repr(r2[0]) + ',' + repr(r2[1]) + ') = ' + repr(psi[ind_x2,ind_y2]))

            #print('Current \psi(0,0) = ' + repr(psi[0,0]))
            #print('Current \psi(0,Ny) = ' + repr(psi[0,int(Ny-1)]))
            #

            psi_tmp = psi / np.max(abs(psi))
            #
            if dflag == True:
                #
                if len(psix_dat) == 0:
                    #
                    psix_dat = np.append(psix_dat,psi_tmp[:,int(Ny/2)].reshape(Nx),axis=0)
                    psiy_dat = np.append(psiy_dat,psi_tmp[int(Nx/2),:].reshape(Ny),axis=0)
                #
                elif len(psix_dat.ravel()) == Nx:
                    #
                    psix_dat = np.append([psix_dat],[psi_tmp[:,int(Ny/2)].reshape(Nx)],axis=0)
                    psiy_dat = np.append([psiy_dat],[psi_tmp[int(Nx/2),:].reshape(Ny)],axis=0)
                #
                else:
                    #
                    psix_dat = np.append(psix_dat,[psi_tmp[:,int(Ny/2)].reshape(Nx)],axis=0)
                    psiy_dat = np.append(psiy_dat,[psi_tmp[int(Nx/2),:].reshape(Ny)],axis=0)
                #
            #
        #
        if la.norm(dpsi,2) <= tol and c >= cmin:
            #
            run_flag = False
        #
        if c >= cmax:
            #
            run_flag = False
            print('Reached maximum iteration number!')
        #
    #
    print('------------------------')
    #print('initial n0: ' + repr(n0) + ', final n0: ' + repr(np.abs(psi[0,0])**2))
    print('Final frob. norm: ' + repr(la.norm(dpsi,2)))
    print(repr(c) + ' iterations executed')
    #
    return x, y, psi0, psi, fn_dat, psix_dat, psiy_dat, j_in, j0, j1, p_ri, p_ir, k_dd, r_in, ind_x1, ind_y1, ind_x2, ind_y2
#

Dx = (7.,0.1)
Dy = (7.,0.1)
Dt = 5e-5
tol = 1e-9
cpar = [10,200,10] #min its., max its., sampling
bcg_tol = 1e-10
typ = 0 # initial guess: Flat (0) Gaussian (1)
tau = 4/7.
g = 750. # interaction strengh N * a_s / a_z
N = 1.0
r1 = [2,-1.]
r2 = [-2,1.]
dflag = False # toggle diagnostics
pflag = True # toggle v pin
T = 0

R0 = 6.0
V0 = 5e2

#

t0 = time.time()
x, y, psi0, psi, fn_dat, psix_dat, psiy_dat, j_in, j0, j1, p_ri, p_ir, k_dd, r_in, ind_x1, ind_y1, ind_x2, ind_y2 = do_pri(Dx=Dx,Dy=Dy,r1=r1,r2=r2,g=g,N=N,R0=R0,V0=V0,tol=tol,cpar=cpar,bcg_tol=bcg_tol,tau=tau,typ=typ,dflag=dflag,pflag=pflag)
#_, sptm, ts = rltm(psi_in=psi,Dx=Dx,Dy=Dy,Dt=Dt,g=g,R0=R0,V0=V0,T=T)
#anim(Dx=Dx,Dy=Dy,gnd=psi,sptm=sptm,T=T)

#

print('Total time: ' + repr(time.time() - t0))

#

f1, ax = plt.subplots(3,2,figsize=(6,7))
#
dz = ax[0,0].pcolor(x,y,np.abs(psi0.T)**2,cmap='Blues_r')
pz = ax[0,1].pcolor(x,y,np.arctan2(np.imag(psi0.T),np.real(psi0.T)),cmap='Blues_r')
ax[0,0].set_title(r'$|\psi_0(x,y)|^2$')
ax[0,1].set_title(r'$\phi[\psi_0(x,y)]$')
ax[0,0].set_xlabel(r'$x$')
ax[0,0].set_ylabel(r'$y$')
ax[0,1].set_xlabel(r'$x$')
ax[0,1].set_ylabel(r'$y$')
ax[0,0].axis('equal')
ax[0,1].axis('equal')
f1.colorbar(dz,ax=ax[0,0])
f1.colorbar(pz,ax=ax[0,1])
#
df = ax[1,0].pcolor(x,y,np.abs(psi.T)**2,cmap='Blues_r') #,vmin=0.,vmax=np.max(np.abs(psi)**2),cmap='Blues_r')
pf = ax[1,1].pcolor(x,y,np.arctan2(np.imag(psi.T),np.real(psi.T)),cmap='Blues_r')
ax[1,0].plot(r1[0],r1[1],'o',markerfacecolor='none',color='tab:red')
ax[1,0].plot(r2[0],r2[1],'o',markerfacecolor='none',color='tab:red')
ax[1,1].plot(r1[0],r1[1],'or',markerfacecolor='none')
ax[1,1].plot(r2[0],r2[1],'or',markerfacecolor='none')
ax[1,0].set_title(r'$|\psi(x,y)|^2$')
ax[1,1].set_title(r'$\phi[\psi(x,y)]$')
ax[1,0].set_xlabel(r'$x$')
ax[1,0].set_ylabel(r'$y$')
ax[1,1].set_xlabel(r'$x$')
ax[1,1].set_ylabel(r'$y$')
ax[1,0].axis('equal')
ax[1,1].axis('equal')
f1.colorbar(df,ax=ax[1,0])
f1.colorbar(pf,ax=ax[1,1])
#
ax[2,0].plot(x,np.abs(psi0[:,ind_y1])**2,color='tab:green')
a2 = ax[2,0].twinx()
a2.plot(x,np.abs(psi[:,ind_y1])**2)
a2.plot(y,np.abs(psi[ind_x1,:])**2,'--')
a2.set_ylabel(r'$|\psi(x,$' + repr(int(r1[1])) + r'$)|^2,\ |\psi($' + repr(int(r1[0])) + r'$,y)|^2$')
#a2.plot(x,np.abs(get_ini(Dx=Dx,Dy=Dy,n0=n0,g=g,typ=0)[:,int(2*Dy[0]/Dy[1]+1)/2])**2,'--')
ax[2,0].set_xlim([-Dx[0],Dx[0]])
ax[2,0].set_xlabel(r'$x$')
ax[2,0].set_ylabel(r'$|\psi_0(x,$' + repr(int(r1[1])) + r'$)|^2$')
a2.set_xlim([-Dx[0],Dx[0]])
ax[2,1].semilogy(fn_dat,'+')
ax[2,1].set_xlabel(r'${\rm iterations}$')
ax[2,1].set_ylabel(r'$||\delta\psi||$')
plt.tight_layout()
f1.show()

#
if dflag == True:
    #
    f2, ax = plt.subplots(4,1,figsize=(5,8))
    #
    xr = ax[0].pcolor(np.linspace(1,psix_dat.shape[0],psix_dat.shape[0]),x,np.real(psix_dat.T),cmap='Blues_r')
    xi = ax[1].pcolor(np.linspace(1,psix_dat.shape[0],psix_dat.shape[0]),x,np.imag(psix_dat.T),cmap='Blues_r')
    ax[0].set_ylabel(r'$x$')
    ax[1].set_ylabel(r'$x$')
    ax[0].set_title(r'${\rm Re}(\psi(x,0))$')
    ax[1].set_title(r'${\rm Im}(\psi(x,0))$')
    #
    yr = ax[2].pcolor(np.linspace(1,psiy_dat.shape[0],psiy_dat.shape[0]),y,np.real(psiy_dat.T),cmap='Blues_r')
    yi = ax[3].pcolor(np.linspace(1,psiy_dat.shape[0],psiy_dat.shape[0]),y,np.imag(psiy_dat.T),cmap='Blues_r')
    ax[2].set_ylabel(r'$y$')
    ax[3].set_ylabel(r'$y$')
    ax[2].set_title(r'${\rm Re}(\psi(0,y))$')
    ax[3].set_title(r'${\rm Im}(\psi(0,y))$')
    ax[3].set_xlabel(r'${\rm iterations}$')
    #
    f2.colorbar(xr,ax=ax[0])
    f2.colorbar(xi,ax=ax[1])
    #
    f2.colorbar(yr,ax=ax[2])
    f2.colorbar(yi,ax=ax[3])
    #
    plt.tight_layout()
    f2.show()
#

#
