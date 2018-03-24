import numpy as np
from scipy.signal import lfilter
from six.moves import xrange

def beamform_filter_time(s, h, zi):

# This function beaform filters the signals into 1 clean signal
#
# Input:
# s - input speech matrix. Each column is a channel
# h - filtering coefficients. Each column is a channel
# zi - initial condition. Each column is a channel
#
# Ouput:
# y - beamformed signal
# zf - final condition. Each column is a channel

    
    if zi == 0:
        zi = np.zeros((h.shape[0]-1, h.shape[1]), dtype=h.dtype)
        
    zf = np.zeros_like(zi) 
    s_filtered = np.zeros_like(s)
    num_channels = s.shape[1]
    
    # filter channel by channel
    for channel_idx in xrange(num_channels):
        s_filtered[:, channel_idx], zf[:, channel_idx] \
        = lfilter(h[:, channel_idx], 1, s[:, channel_idx], 0, zi[:, channel_idx])
        
    y = np.sum(s_filtered, axis=1) 
    
    return (y, zf)
    


