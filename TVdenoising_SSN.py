# HUBER TV denoising using Semismooth Newton method on the primal-dual
# optimality conditions


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.sparse import diags, hstack, vstack, identity
from scipy.sparse.linalg import spsolve
from scipy import sparse
from scipy.linalg import norm

def diff_operators(n,m):

    DX=diags([np.hstack((-np.ones(n*(m-1)), np.zeros(n))), np.ones(n*(m-1))], offsets=[0, n])
    DY=diags([np.tile(np.hstack((-np.ones(n-1), [0])),m), np.hstack((np.tile(np.hstack((np.ones(n-1), [0])),m-1),np.ones(n-1)))], offsets=[0,1])

    divX = -DX.transpose()
    divY = -DY.transpose()

    en13 = np.hstack(([-2], -3 * np.ones(m - 2), [-2]))
    en2 = np.hstack((-3*np.ones((n-2,1)), -4*np.ones((n-2,m-2)),-3*np.ones((n-2,1))))
    En=np.vstack((en13, en2, en13))
    en= En.reshape( n*m, order='F')
    e= np.ones(n*(m-1))
    e1= np.hstack((np.tile(np.hstack((np.ones(n-1), [0])),m-1),np.ones(n-1)))

    Lapn= diags([e, e1, en, e1, e], offsets=[-n , -1, 0, 1, n])

    return DX, DY, divX, divY, Lapn

def reproject(p, alpha):
    y1=np.zeros(p.shape)

    y1[0,:,:] = p[0,:,:]/np.maximum(np.sqrt(p[0,:,:]**2 + p[1,:,:]**2)/alpha, 1)
    y1[1,:,:] = p[1,:,:]/np.maximum(np.sqrt(p[0,:,:]**2 + p[1,:,:]**2)/alpha, 1)

    return y1

def residual_optimality(u, p, f, alpha, gamma):

    (n,m)=u.shape

    p1=p[0,:,:]
    p2=p[1,:,:]

    dxm_p1 = np.hstack(( p1[:,0:m-1] , np.zeros((n,1)) )) - np.hstack(( np.zeros((n,1))  , p1[:,0:m-1]  ))
    dxm_p2 = np.vstack((p2[0:n-1, :], np.zeros((1, m)))) - np.vstack((np.zeros((1, m)), p2[0:n-1, :]))

    #dxm_dxp = np.hstack(( dxp[:,0:m] , np.zeros(n,1)  )) - np.hstack(( np.zeros(n,1)  , dxp[:,0:m]  ))
    #dym_dyp = np.vstack((dxp[0:n, :], np.zeros(1, m))) - np.vstack((np.zeros(1, m), dxp[0:n, :]))

    resu= norm(dxm_p1 + dxm_p2 + f - u, 'fro')

    ux = np.hstack((u[:, 1:], u[:, [m - 1]])) - u
    uy = np.vstack((u[1:, :], u[[n - 1], :])) - u
    um = np.maximum(np.sqrt(ux**2 + uy**2) , gamma)

    h1= um*p[0,:,:] - alpha*ux
    h2= um*p[1,:,:] - alpha*uy

    resp = norm(np.vstack(( h1.reshape( n*m, order='F')  , h2.reshape( n*m, order='F') )), 'fro')

    return resu, resp

#mat_contents = sio.loadmat('parrot')
#clean=mat_contents['parrot']
#f=mat_contents['parrot_noisy_01']

def TV_denoising(f,clean, alpha, gamma):

    (n,m) = clean.shape
    gamma_mtx = gamma*np.ones((n,m))
    alpha_mtx = alpha*np.ones((n,m))

    (DX, DY, divX, divY, Lapn)=diff_operators(n,m)
    B=sparse.eye(n*m,n*m)


    # Initializations

    u = np.zeros((n,m))
    p = np.zeros((2, n, m))
    bp = np.zeros((2, n, m))

    u_adj = np.zeros((n,m))
    p_adj = np.zeros((2, n, m))

    ux = np.reshape(DX@u.ravel(order='F'), (n,m), order='F')
    uy = np.reshape(DY@u.ravel(order='F'), (n,m), order='F')
    um = np.maximum(np.sqrt(ux**2 + uy**2), gamma_mtx)

    p[0, :, :] = alpha_mtx*(ux/um)
    p[1, :, :] = alpha_mtx*(uy/um)

    act_u = np.zeros((n,m))
    uxm   = np.zeros((n,m))
    uym   = np.zeros((n,m))

    max_newton_its = 20
    tol_g = 1e-1

    for k in range(0,max_newton_its):

         # Calculate right hand side for Newton linear system

         ux = np.reshape(DX @ u.ravel(order='F'), (n, m), order='F')
         uy = np.reshape(DY @ u.ravel(order='F'), (n, m), order='F')
         ut = np.sqrt(ux ** 2 + uy ** 2)
         um = np.maximum(ut, gamma_mtx)

         act_u[ut>=gamma_mtx] = 1
         act_u[ut< gamma_mtx] = 0

         index=np.nonzero(ut>=gamma_mtx)
         uxm[index] = ux[index]/ut[index]
         uym[index] = uy[index]/ut[index]

         uu = act_u*(uxm*ux + uym*uy)/um

         bp[0, :, :] = uu*p[0, :, :]
         bp[1, :, :] = uu*p[1, :, :]

         b = f.ravel(order='F') + hstack((divX, divY))@np.hstack((np.ravel(bp[0,:,:], order='F') ,np.ravel(bp[1,:,:], order='F')))

         # Build matrix for Newton linear system

         A_der = B - hstack((divX, divY))@diags([np.hstack((alpha_mtx.ravel(order='F'), alpha_mtx.ravel(order='F')))], offsets=[0])@diags([np.hstack((1/(um.ravel(order='F')), 1/(um.ravel(order='F')))) ], offsets=[0])@vstack((DX, DY))
         A_der = A_der + divX@diags([1/(um.ravel(order='F'))], offsets=[0])@diags([np.ravel(p[0,:,:], order='F')], offsets=[0])@diags([act_u.ravel(order='F')*uxm.ravel(order='F')], offsets=[0])@DX
         A_der = A_der + divX@diags([1/(um.ravel(order='F'))], offsets=[0])@diags([np.ravel(p[0,:,:], order='F')], offsets=[0])@diags([act_u.ravel(order='F')*uym.ravel(order='F')], offsets=[0])@DY
         A_der = A_der + divY@diags([1/(um.ravel(order='F'))], offsets=[0])@diags([np.ravel(p[1,:,:], order='F')], offsets=[0])@diags([act_u.ravel(order='F')*uxm.ravel(order='F')], offsets=[0])@DX
         A_der = A_der + divY@diags([1/(um.ravel(order='F'))], offsets=[0])@diags([np.ravel(p[1,:,:], order='F')], offsets=[0])@diags([act_u.ravel(order='F')*uym.ravel(order='F')], offsets=[0])@DY

         uold = u.copy()
         pold = p.copy()

         u_col = spsolve(A_der, b)

         u = np.reshape(u_col, (n, m), order='F')
    
         # Calculate new p

         p_vec = diags([np.hstack((alpha_mtx.ravel(order='F'), alpha_mtx.ravel(order='F')))], offsets=[0])\
             @diags([np.hstack((1/(um.ravel(order='F')), 1/(um.ravel(order='F')))) ], offsets=[0])\
             @vstack((DX, DY))@u.ravel(order='F')
         p_vec = p_vec - diags([np.hstack((np.ravel(pold[0,:,:], order='F') ,np.ravel(pold[1,:,:], order='F')))], offsets=[0])\
             @vstack((identity(n*m), identity(n*m)))@diags([ 1/(um.ravel(order='F')) ], offsets=[0])\
             @diags([act_u.ravel(order='F')], offsets=[0])\
             @(diags([uxm.ravel(order='F')], offsets=[0])@DX@u.ravel(order='F')  +  diags([uym.ravel(order='F')], offsets=[0])@DY@u.ravel(order='F'))
         p_vec = p_vec + diags([np.hstack((np.ravel(pold[0,:,:], order='F') ,np.ravel(pold[1,:,:], order='F')))], offsets=[0])\
             @vstack((identity(n*m), identity(n*m)))@diags([ 1/(um.ravel(order='F')) ], offsets=[0])\
             @diags([act_u.ravel(order='F')], offsets=[0])\
             @(diags([uxm.ravel(order='F')], offsets=[0])@ux.ravel(order='F') + diags([uym.ravel(order='F')], offsets=[0])@uy.ravel(order='F') )

         p[0, :, :] = np.reshape(p_vec[0:(n*m)], (n, m), order='F')
         p[1, :, :] = np.reshape(p_vec[(n*m):], (n, m), order='F')

         p = reproject(p, alpha_mtx)
        
         (resu, resp)=residual_optimality(u, p, f, alpha_mtx, gamma_mtx)

         print('Newton iteration :', k+1, 'The residuals of u and p are', resu, 'and ', resp)

         plt.figure(figsize = (7,7)) 
         imgplot2 = plt.imshow(u)
         imgplot2.set_cmap('gray')
         plt.pause(0.05)


         act_u = np.zeros((n,m))
         uxm   = np.zeros((n,m))
         uym   = np.zeros((n,m))
    
         if resu<tol_g and resp<tol_g:
             print('The Newton method converged')
             break
            
    return u






















