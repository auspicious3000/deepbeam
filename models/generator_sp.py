import numpy as np
import tensorflow as tf
import bawn
from utils import mu_law_bins, random_bins


class Model_Simple(object):

    def __init__(self,
                 len_input_noisy=24570,
                 len_output=16384,
                 num_blocks_noisy=4, 
                 num_layers_noisy=10,
                 num_classes=256, 
                 num_post_layers=2,        #number of dense layers
                 num_residual_channels_noisy=64,           
                 num_skip_channels=256):
        
        _, self.bins_center = mu_law_bins(bawn.NUM_CLASSES) 
        self.bins_center_sq = self.bins_center ** 2
        
        self.len_output = len_output       
        self.num_classes = num_classes
                
        self.num_blocks_noisy = num_blocks_noisy
        self.num_layers_noisy = num_layers_noisy
        
        self.num_post_layers = num_post_layers
        
        self.num_residual_channels_noisy = num_residual_channels_noisy
        
        self.num_skip_channels = num_skip_channels
        
        
        inputs_noisy = tf.placeholder(tf.float32, shape=(None, len_input_noisy))
        
        self.inputs_noisy = inputs_noisy
        
                
        _, skips_noisy_batch = bawn._wavnet(inputs=inputs_noisy,
                                            num_blocks=num_blocks_noisy, 
                                            num_layers=num_layers_noisy, 
                                            num_residual_channels=num_residual_channels_noisy, 
                                            num_skip_channels=num_skip_channels, 
                                            len_output=len_output, 
                                            filter_width=3,
                                            speech_type='noisy',
                                            bias=True,
                                            trainable=False)
                      
                        
        outputs_ll_batch = bawn._post_processing(skips_noisy_batch, 
                                                 num_post_layers, 
                                                 num_classes, 
                                                 'noisy/',
                                                 trainable=False)
        
               
        self.outputs_softmax_batch = tf.nn.softmax(outputs_ll_batch, axis=1)
        self.skips_noisy_batch = tf.add_n(skips_noisy_batch)
        self.outputs_ll_batch = outputs_ll_batch
       
        
        #params of batch training model
        self.saver = tf.train.Saver(tf.global_variables())
        
        
        # Create an ExponentialMovingAverage object
        ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        real_var = [v for v in tf.global_variables()]
        shadow_name = []
        for v in tf.global_variables():
            if 'noisy' in v.name:
                shadow_name.append(v.op.name)
            else:
                shadow_name.append(ema.average_name(v))
                
               
        self.saver_shadow = tf.train.Saver(dict(zip(shadow_name, real_var)))
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement=True
        config.log_device_placement=True
        config.intra_op_parallelism_threads=16
        config.inter_op_parallelism_threads=4
        self.sess = tf.Session(config=config)
        
        
    def run_offline(self, inputs_noisy):
        feed_dict = {self.inputs_noisy: inputs_noisy}
        output_dist = self.sess.run(self.outputs_softmax_batch, feed_dict=feed_dict)
        
        # predictions
        indices = np.argmax(output_dist, axis=1)
        predictions = np.array(self.bins_center[indices])
                      
        pred_mean = np.matmul(self.bins_center, output_dist[0])
        pred_sq_mean = np.matmul(self.bins_center_sq, output_dist[0])
        pred_var = np.maximum(1.0/5000, pred_sq_mean - pred_mean ** 2)
        weights = 1.0/pred_var
        
        return (predictions, weights)    
    
    
    
