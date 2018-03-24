import numpy as np
from scipy.linalg import toeplitz, lstsq
from six.moves import xrange
import matplotlib.pyplot as plt

def shift_bit_length(x):
    #be careful about x==0
    return 1<<(x-1).bit_length()


def update_beamform_coef(r, e, h_order):
# This function updates the beamforming coefficients using Weiner filtering
#
# Input:
# r - Noisy observation. Each column is a channel
# e - target clean speech
# h_order - the order of the beamforming coefficients
#
# Output:
# h - beamforming coefficients. Each column corresponds to a channel.

    num_channels = r.shape[1]
    
    S = np.fft.rfft(r, n=shift_bit_length(r.shape[0]+h_order), axis=0)
        
    # fill the autocorrelation matrix
    R = np.zeros((num_channels*h_order, num_channels*h_order))
    for channel_idx1 in xrange(num_channels):
        for channel_idx2 in xrange(channel_idx1+1):
            autocorr_temp = np.fft.irfft(np.conjugate(S[:,channel_idx1])*S[:,channel_idx2])
            R_temp = toeplitz(autocorr_temp[:h_order], \
                              np.concatenate(([autocorr_temp[0]], autocorr_temp[-1:-h_order:-1])))
            R[channel_idx1*h_order : (channel_idx1+1)*h_order, \
              channel_idx2*h_order : (channel_idx2+1)*h_order] = R_temp
            if channel_idx1 != channel_idx2:
                R[channel_idx2*h_order : (channel_idx2+1)*h_order, \
                  channel_idx1*h_order : (channel_idx1+1)*h_order] = np.transpose(R_temp)
    
    # cross correlation vector
    E = np.fft.rfft(e, n=shift_bit_length(e.shape[0]+h_order))
    xcorr_temp = np.fft.irfft(np.conjugate(S)*E[:,None], axis=0)
    xcorr_temp = xcorr_temp[:h_order, :]
    p = xcorr_temp.flatten(order='F')
    
    # recondition R
    # R = R + np.diag(1.0 * np.diag(R), 0)
    #R = np.diag(1.0 * np.diag(R), 0)
    
    # compute the beamforming coefficients
    h_temp, _, rank = np.linalg.lstsq(R,p)[0:3]
    #plt.plot(h_temp)
    #plt.show()
    h = h_temp.reshape((h_order, num_channels), order='F')
    
    return (R, p, h)          