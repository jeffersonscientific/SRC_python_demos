import random
import numpy
import numba
import multiprocessing as mpp
#
#@numba.jit()
#def mandelbrotter(re_min=-2.0, im_min=-1.5, re_max=1.0, im_max=1.5, d_re=.01, d_im=.01, max_iter=256):
def mandelbrotter(re_min=-2.0, im_min=-1.5, re_max=1.0, im_max=1.5, N_x=512, N_y=512, max_iter=256):
#    max_iter = 256
#    width = 256
#    height = 256
#    center = -0.8+0.0j
#    extent = 3.0+3.0j
#    scale = max((extent / width).real, (extent / height).imag)j
    # if given d_re, d_im
    #N_x = int((re_max - re_min)/d_re)
    #N_y = int((im_max - im_min)/d_im)
    d_re = (re_max - re_min)/N_x
    d_im = (im_max - im_min)/N_y
    #
    print('** DEBUG: ', N_x, N_y)
    print('** DEBUG: ', d_re, d_im)
    #
    Ks = numpy.zeros((N_x, N_y), int)
    Zs = numpy.zeros((N_x, N_y), float)
    #Ks = [[0 for k in range(N_x)] for j in range(N_y)]
    #Zs = [[0. for k in range(N_x)] for j in range(N_y)]
    #
    for j in range(N_y):
        for i in range(N_x):
            #c = center + (i - width // 2 + (j - height // 2)*1j) * scale
            c = re_min + d_re*i + 1j*(im_min + d_im*j)
            z = 0. + 0j
            #
            k = 0
            #for k in range(max_iter):
            while k<max_iter and (z*z.conjugate()).real < 4.0:
                #z = z**2 + c
                z = z*z + c
                k+=1
#                if (z * z.conjugate()).real > 4.0:
#                    break
            Ks[j, i] = k
            Zs[j, i] = (z*z.conjugate()).real
    #
    return {'Ks': Ks, 'Zs':Zs}
      
