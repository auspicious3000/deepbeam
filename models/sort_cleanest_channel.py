import numpy as np
from scipy.stats.mstats import mquantiles
from six.moves import xrange

def sort_cleanest_channel(s):
# This function finds the cleanest channel by computing the lower quantiles
#
# Input:
# s - each column is a channel of signal
#
# Output:
# cleanest_channel - the index of the estimated cleanest channel
    num_channels = s.shape[1]
    
    # compute the quatiles of each channel
    q = mquantiles(np.absolute(s), 0.4, alphap=0.5, betap=0.5, axis=0)
        
    # compute quantile energy ratios
    qr = np.zeros([num_channels, 1])
    for channel_id in xrange(num_channels):
        qr[channel_id, :] = np.linalg.norm(s[np.absolute(s[:, channel_id]) < q[:,channel_id], \
                                          channel_id]) / np.linalg.norm(s[:, channel_id])
        
    sorted_channel = np.argsort(qr, axis=0) 
    
    return sorted_channel[:,0]