# Deepbeam
Deep learning based Speech Beamforming

## Requirements
tensorflow, scipy, fftw, h5py

## Train Wavenet-based enhancement model
Noisy input data filename: noisy_train.mat  
Dimension：                [24570, NUM_TOKENS]
Content:                   noisy waveforms

Clean ouput data filename: target_train.mat
Dimension:                 [16384, NUM_TOKENS]
Content:                   256 mu-law quantized bin index of clean waveforms

The above become numpy arrays after loaded into python,
you can generate your own traning data and modify the model architecture accordingly.

To train the enhancement model, place the data in the same directory as the training code,
then execute the following:

python bawn_sp_multi_gpu_train_v2.py /logdir NUM_GPUS
