#
# Newton-Rahpson-Richardson iterator
# 2D homogeneous vortex
#

import numpy as np
import scipy as sp
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
    x = np.linspace(-(Dx[0]-0.0),Dx[0]+0.0,Nx)
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
def get_ini(Dx,Dy,r0,g,N,typ):
    #
    Nx = int(2*Dx[0]/Dx[1] + 1)
    Ny = int(2*Dy[0]/Dy[1] + 1)
    #
    (dx, dy) = (Dx[1], Dy[1])
    #
    x = np.linspace(-(Dx[0]-0.0),Dx[0]+0.0,Nx)
    y = np.linspace(-Dy[0],Dy[0],Ny)

    xx, yy = np.meshgrid(x,y,indexing='ij')
    #
    psi_out = np.zeros((Nx,Ny),dtype='complex')
    #
    if typ == 0:
        #
        psi = np.tanh(2.0*pow(pow(xx-r0[0],2) + pow(yy-r0[1],2),.5)) # + np.tanh(2*pow(pow(xx+q0,2) + pow(yy,2),.5))
        phi = np.arctan2(yy-r0[1],xx-r0[0]) # - np.arctan2(yy,xx+q0)
        #
        psi_out = pow(psi,2) * np.exp(-1j*phi)
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
def get_mu(Dx,Dy,g,psi_in):
    #
    Nx = int(2*Dx[0]/Dx[1] + 1)
    Ny = int(2*Dy[0]/Dy[1] + 1)
    #
    (dx, dy) = (Dx[1], Dy[1])
    #
    x = np.linspace(-(Dx[0]-0.0),Dx[0]+0.0,Nx)
    y = np.linspace(-Dy[0],Dy[0],Ny)
    #
    ke = 0.
    #
    en = 0.
    vdw = 0.
    #
    xx, yy = np.meshgrid(x,y,indexing='ij')
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
            vdw = vdw + g*dx*dy*np.abs(psi_in[jj,kk])**4
            #
        #
    #
    nm = dx*dy*np.trapz(np.trapz(np.abs(psi_in)**2))
    mu = (ke + vdw)/nm
    #
    return mu
#

#
def get_ind(Dx,Dy,r0):
    #
    Nx = int(2*Dx[0]/Dx[1] + 1)
    Ny = int(2*Dy[0]/Dy[1] + 1)
    #
    x = np.linspace(-(Dx[0]-0.0),Dx[0]+0.0,Nx)
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
def get_res(Dx,Dy,r0,g,mu,psi_in,pflag):
    #

    Nx = int(2*Dx[0]/Dx[1] + 1)
    Ny = int(2*Dy[0]/Dy[1] + 1)
    #

    x = np.linspace(-(Dx[0]-0.0),Dx[0]+0.0,Nx)
    y = np.linspace(-Dy[0],Dy[0],Ny)
    #

    d2 = set_sp(Dx=Dx,Dy=Dy)
    r = np.zeros((2*Nx*Ny,1),dtype='float')
    #

    k_rr = -0.5*d2*np.real(psi_in).ravel()
    k_ii = -0.5*d2*np.imag(psi_in).ravel()
    #

    p_rr = ((g*(pow(np.real(psi_in),2) + pow(np.imag(psi_in),2)) - mu)*np.real(psi_in)).ravel()
    p_ii = ((g*(pow(np.real(psi_in),2) + pow(np.imag(psi_in),2)) - mu)*np.imag(psi_in)).ravel()
    #

    r[:Nx*Ny,0] = k_rr + p_rr
    r[Nx*Ny:,0] = k_ii + p_ii

    if pflag == True:
        #
        r_tmp = r[:Nx*Ny,0].reshape(Nx,Ny)
        i_tmp = r[Nx*Ny:,0].reshape(Nx,Ny)

        #
        ind_x, ind_y = get_ind(Dx=Dx,Dy=Dy,r0=r0)

        r_tmp[ind_x,ind_y] = 0.
        i_tmp[ind_x,ind_y] = 0.

        r[:Nx*Ny,0] = r_tmp.reshape(Nx*Ny)
        r[Nx*Ny:,0] = i_tmp.reshape(Nx*Ny)
    #

    return r
#

#
def get_jac(Dx,Dy,r0,g,mu,psi_in,pflag):
    #

    Nx = int(2*Dx[0]/Dx[1] + 1)
    Ny = int(2*Dy[0]/Dy[1] + 1)
    NN = int(Nx*Ny)
    #
    x = np.linspace(-(Dx[0]-0.0),Dx[0]+0.0,Nx)
    y = np.linspace(-Dy[0],Dy[0],Ny)
    #
    s = np.zeros(4,dtype=int)
    #

    d2 = set_sp(Dx=Dx,Dy=Dy)
    #

    k_dd = -0.5*d2
    #

    p_rr = sparse.diags((g*(3*pow(np.real(psi_in),2) + pow(np.imag(psi_in),2)) - mu).ravel(),0,format='csr')
    p_ii = sparse.diags((g*(pow(np.real(psi_in),2) + 3*pow(np.imag(psi_in),2)) - mu).ravel(),0,format='csr')
    #

    p_ri = sparse.diags((2*g*np.real(psi_in)*np.imag(psi_in)).ravel(),0,format='csr')
    p_ir = sparse.diags((2*g*np.imag(psi_in)*np.real(psi_in)).ravel(),0,format='csr')
    #

    j = bmat([[k_dd+p_rr,p_ri],[p_ir,k_dd+p_ii]],format='csr')
    #

    if pflag == True:
        #
        ind_y, ind_x = get_ind(Dx=Dx,Dy=Dy,r0=r0)

        ind = ind_x + ind_y*Ny
        #

        r1 = j.getrow(ind).data.shape[0]
        #
        i1 = np.where(j.getrow(ind).data == j.toarray()[ind,ind])[0][0]

        d1 = np.where(j.data == j.toarray()[ind,ind])[0][0]

        v1 = j.data[d1]

        #
        #if r1 == 6:
            #
            #print('hi')
            #i2 = np.where(j.getrow(ind).data == j.toarray()[ind,ind+NN])[0][0]

            #d2 = np.where(j.data == j.toarray()[ind,ind+NN])[0][0]

            #v2 = j.data[d2]
        #

        s1 = int(d1 - i1)
        s2 = int(s1 + r1)

        for jj in range(s1,s2,1):
            #
            j.data[jj] = 0.
        #
        r2 = j.getrow(ind+NN).data.shape[0]

        #
        i4 = np.where(j.getrow(ind+NN).data == j.toarray()[ind+NN,ind+NN])[0][0]

        d4 = np.where(j.data == j.toarray()[ind+NN,ind+NN])[0][0]

        v4 = j.data[d4]
        #
        #if r2 == 6:
            #
            #print('hi')
            #i3 = np.where(j.getrow(ind+NN).data == j.toarray()[ind+NN,ind])[0][0]

            #d3 = np.where(j.data == j.toarray()[ind+NN,ind])[0][0]

            #v3 = j.data[d3]
        #

        s3 = int(d4 - i4)
        s4 = int(s3 + r2)

        for jj in range(s3,s4,1):
            #
            j.data[jj] = 0.
        #
        j.data[d1] = v1

        #
        #if r1 == 6:
            #
            #j.data[d2] = v2
        #
        j.data[d4] = v4

        #
        #if r2 == 6:
            #
            #j.data[d3] = v3
        #
        s[0] = s1; s[1] = s2; s[2] = s3; s[3] = s4;
    #
    return j, s
#

#
def do_pri(Dx,Dy,r0,g,N,tol,cpar,bcg_tol,tau,typ,dflag,pflag):
    #
    Nx = int(2*Dx[0]/Dx[1] + 1)
    Ny = int(2*Dy[0]/Dy[1] + 1)

    #
    (dx, dy) = (Dx[1], Dy[1])

    #
    x = np.linspace(-(Dx[0]-0.0),Dx[0]+0.0,Nx)
    y = np.linspace(-Dy[0],Dy[0],Ny)

    #
    psi0 = get_ini(Dx=Dx,Dy=Dy,r0=r0,g=g,N=N,typ=typ)
    mu0 = get_mu(Dx=Dx,Dy=Dy,g=g,psi_in=psi0)

    psi = psi0
    mu = mu0

    #
    ind_x, ind_y = get_ind(Dx=Dx,Dy=Dy,r0=r0)

    psi[ind_x,ind_y] = 0.

    #N0 = dx*dy*np.trapz(np.trapz(np.abs(psi0)**2))
    #n0 = N0 / (2*Dx[0]*2*Dy[0])

    #
    print('\psi initial norm: ' + repr(dx*dy*np.trapz(np.trapz(np.abs(psi0)**2))))
    #

    print('Numerical mu: ' + repr(mu0))
    print('Analytical mu: ' + repr(mu))
    print('Initial \psi(' + repr(r0[0]) + ',' + repr(r0[1]) + ') = ' + repr(psi[ind_x,ind_y]))

    print('Initial \psi(-Lx,Ly) = ' + repr(psi[0,int(Ny-1)]))
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
        j_in, s = get_jac(Dx=Dx,Dy=Dy,r0=r0,g=g,mu=mu,psi_in=psi,pflag=pflag)
        r_in = get_res(Dx=Dx,Dy=Dy,r0=r0,g=g,mu=mu,psi_in=psi,pflag=pflag)

        #
        t0 = time.time()
        dpsi, ec = bicgstab(j_in, -tau*r_in, tol=bcg_tol)
        print('bcgstab time: ' + repr(time.time() - t0))
        #
        psi = psi + dpsi[:Nx*Ny].reshape(Nx,Ny) + 1j*dpsi[Nx*Ny:].reshape(Nx,Ny)

        print('Current \psi(' + repr(r0[0]) + ',' + repr(r0[1]) + ') = ' + repr(psi[ind_x,ind_y]))

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

            print('Current \psi(-Lx,-Ly) = ' + repr(psi[0,0]))
            print('Current \psi(-Lx,Ly) = ' + repr(psi[0,int(Ny-1)]))
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
    return x, y, psi0, psi, fn_dat, psix_dat, psiy_dat, j_in, r_in, ind_x, ind_y, s
#

Dx = (4.,0.1)
Dy = (4.,0.1)
Dt = 5e-5
tol = 1e-8
cpar = [10,50,10] #min its., max its., sampling
bcg_tol = 1e-9
typ = 0 # initial guess: Flat (0) Gaussian (1)
tau = 4/7.
g = 2000. # interaction strengh N * a_s / a_z
N = 1.0
r0 = [1.,1.]
dflag = True # toggle diagnostics
pflag = True # toggle v pin
T = 0

#

t0 = time.time()
x, y, psi0, psi, fn_dat, psix_dat, psiy_dat, j_in, r_in, ind_x, ind_y, s = do_pri(Dx=Dx,Dy=Dy,r0=r0,g=g,N=N,tol=tol,cpar=cpar,bcg_tol=bcg_tol,tau=tau,typ=typ,dflag=dflag,pflag=pflag)
#_, sptm, ts = rltm(psi_in=psi,Dx=Dx,Dy=Dy,Dt=Dt,g=g,T=T)
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
ax[1,0].plot(r0[0],r0[1],'o',markerfacecolor='none',color='tab:red')
ax[1,1].plot(r0[0],r0[1],'or',markerfacecolor='none')
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
ax[2,0].plot(x,np.abs(psi0[:,ind_y])**2,color='tab:green')
a2 = ax[2,0].twinx()
a2.plot(x,np.abs(psi[:,ind_y])**2)
a2.plot(y,np.abs(psi[ind_x,:])**2,'--')
a2.set_ylabel(r'$|\psi(x,$' + repr(int(r0[1])) + r'$)|^2,\ |\psi($' + repr(int(r0[0])) + r'$,y)|^2$')
#a2.plot(x,np.abs(get_ini(Dx=Dx,Dy=Dy,n0=n0,g=g,typ=0)[:,int(2*Dy[0]/Dy[1]+1)/2])**2,'--')
ax[2,0].set_xlim([-Dx[0],Dx[0]])
ax[2,0].set_xlabel(r'$x$')
ax[2,0].set_ylabel(r'$|\psi_0(x,$' + repr(int(r0[1])) + r'$)|^2$')
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
