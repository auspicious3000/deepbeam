import numpy as np
from six.moves import xrange
from pyfftw import interfaces
import matplotlib as plt
from time import time
import cPickle as pickle
import pyfftw
import os
PATH = "../assets/wisdom.p"


def shift_bit_length(x):
    #be careful about x==0
    return 1<<(x-1).bit_length()

def my_circulant(x, tap):
    y = np.zeros((x.shape[0], tap))
    for i in xrange(tap):
        y[:,i] = np.roll(x, i)
    return y

def update_beamform_coef_with_weights(r, e, w, h_order):
    # This function updates the beamforming coefficients using Weiner filtering
    #
    # Input:
    # r - Noisy observation. Each column is a channel
    # e - target clean speech
    # w - weights for each sample 
    # h_order - the order of the beamforming coefficients
    #
    # Output:
    # h - beamforming coefficients. Each column corresponds to a channel.
    interfaces.cache.enable()
    num_channels = r.shape[1]
    
    # pad each channel with zeros so that the circular shift of the autocorrelation is error free
    n = shift_bit_length(r.shape[0]+h_order)
    r = np.lib.pad(r, ((0,n-r.shape[0]),(0,0)), 'constant', constant_values=0.0)  
    e = np.lib.pad(e, (0,n-e.shape[0]), 'constant', constant_values=0.0)
    w = np.lib.pad(w, (0,n-w.shape[0]), 'constant', constant_values=0.0)
    
    S_ = interfaces.numpy_fft.rfft(np.hstack((r,(w*e)[:,None])), axis=0, threads=20)
    S = S_[:,:-1]
    E = S_[:,-1]
    
    # fill the autocorrelation matrix
    R = np.zeros((num_channels*h_order, num_channels*h_order))
    for channel_idx1 in xrange(num_channels):
        y_l = my_circulant(r[:,channel_idx1], h_order)
        wS = interfaces.numpy_fft.rfft((w[:,None]*y_l), axis=0, threads=20)
        for channel_idx2 in xrange(channel_idx1+1):
            # compute yl[]yr[] column-wise
            ylyr = wS*np.conjugate(S[:,channel_idx2:channel_idx2+1])
            xcorr_temp = interfaces.numpy_fft.irfft(ylyr, axis=0, threads=20)
            
            R_temp = xcorr_temp[:h_order,:]
            # fill the R matrix
            R[channel_idx1*h_order : (channel_idx1+1)*h_order, \
              channel_idx2*h_order : (channel_idx2+1)*h_order] = R_temp.T
            if channel_idx1 != channel_idx2:
                R[channel_idx2*h_order : (channel_idx2+1)*h_order, \
                  channel_idx1*h_order : (channel_idx1+1)*h_order] = R_temp
        
        
    # cross correlation vector
    
    #E = interfaces.numpy_fft.rfft(w*e, threads=20)
    xcorr_temp = interfaces.numpy_fft.irfft((np.conjugate(S)*E[:,None]), axis=0, threads=20)
    xcorr_temp = xcorr_temp[:h_order, :]
    p = xcorr_temp.flatten(order='F')
            
    # compute the beamforming coefficients
    h_temp = np.linalg.lstsq(R,p,rcond=None)[0]
    h = h_temp.reshape((h_order, num_channels), order='F')
    
    if not os.path.isfile(PATH) or not os.access(PATH, os.R_OK):
        wisdom = pyfftw.export_wisdom()
        pickle.dump(wisdom, open( "wisdom.p", "wb" ) )
    
    return (R, p, h)        