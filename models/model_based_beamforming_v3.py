import numpy as np
from find_cleanest_channel import find_cleanest_channel
from beamform_filter_time import beamform_filter_time
from update_beamform_coef_with_weights_cp2 import update_beamform_coef_with_weights


len_pad = 4093

def normalize_mat(x, axis=0):
    # normalize by column
    if x.ndim == 1:
        x = x[:,None]
    means = np.mean(x, axis=axis, keepdims=True)
    maximums = np.amax(np.absolute(x), axis=axis, keepdims=True) / 0.99
    x = (x - means) / maximums
    return (x, means, maximums)


def gen_offline(model, x, win_len, pad_len, length):
    
    if x.shape[1] > 270000:
        shift = win_len - 2*pad_len
        num_wins = np.ceil((length-win_len) / float(shift)).astype(np.int32) 
        y = np.zeros((1, num_wins*shift))
        w = np.zeros((num_wins*shift, ))
        for win in xrange(num_wins):
            y[:, win*shift:(win+1)*shift], w[win*shift:(win+1)*shift] \
            = model.run_offline(x[:, win*shift:win*shift+win_len])
    else:
        y, w = model.run_offline(x)
    
    return (y, w)


def model_based_beamforming(model, s, noise, frame_len, len_win, h_order, num_iterations):
    
    num_channels = s.shape[1]
    num_frames = np.floor(s.shape[0] / frame_len).astype(np.int32)
    s_beamformed = np.zeros([s.shape[0], 1])
    noise_beamformed = np.zeros([s.shape[0], 1])   
    
    zi_beamform = 0  # the beamform filter
    zi_noise_beamform = 0  # the beamform filter for noise
    
    # initialize the beamformer
    closest_mic = find_cleanest_channel(s)
    H_init = np.zeros([h_order, num_channels])
    H_init[np.round(2 * h_order / 3), closest_mic] = 1
    SSRs = np.zeros((num_iterations,))
    SNRs = np.zeros((num_iterations,))
    Hs = np.zeros((h_order, num_channels, num_iterations))
    
    # valid part index
    if frame_len > 270000:
        pred_len = (len_win-2*len_pad)*np.ceil(float((frame_len-len_win))/(len_win-2*len_pad)).astype(np.int32)
        id_valid = xrange(len_pad, pred_len+len_pad)
    else:
        id_valid = xrange(len_pad, frame_len-len_pad)
    
    # adaptive beamforming
    for t in xrange(num_frames):
        # set the beamformer to initial value
        if t == 0:
            h = H_init
        print('Frame {}'.format(t))  
        # current index
        current_idx = xrange(t*frame_len, (t+1)*frame_len)  
                       
        for iters in xrange(num_iterations):
            # step 1: obtain the beamformed signal
            s_beamformed_temp, _ = beamform_filter_time(s[current_idx, :], h, zi_beamform)
            
            noise_beamformed_temp, _ = beamform_filter_time(noise[current_idx, :], h, zi_noise_beamform)
                        
            if iters == num_iterations-1:
                continue
            
            # step 2: obtain the denoised signal
            s_beamformed_temp = np.expand_dims(s_beamformed_temp, axis=0)
            pred, w = gen_offline(model, s_beamformed_temp, len_win, len_pad, frame_len)
                        
            # step 4: Weiner filter to update h
            pred,_,_ = normalize_mat(pred, 1)
            _,_,h = update_beamform_coef_with_weights(s[id_valid,:], pred[0,:], w, h_order)
            
        
            print('Iteration {} completed'.format(iters))
            
            
        # apply final filter to produce the output
        s_beamformed[current_idx, 0], _ = beamform_filter_time(s[current_idx, :], h, zi_beamform)  
        noise_beamformed[current_idx, 0], _ = beamform_filter_time(noise[current_idx, :], h, zi_noise_beamform)   
        
               
    return (s_beamformed, noise_beamformed, h)
    