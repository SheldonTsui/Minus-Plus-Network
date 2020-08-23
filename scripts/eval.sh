#!/bin/bash

OPTS=""
OPTS+="--id MUSIC-2mix-LogFreq-resnet18dilated-unet7-linear-frames3stride24-maxpool-binary-weightedLoss-channels16-epoch100-step40_80-225version " 
OPTS+="--list_val ./data/val_wav.csv "
OPTS+="--list_train ./data/train_wav.csv "

# Models
OPTS+="--arch_sound unet7 "
OPTS+="--arch_synthesizer linear "
OPTS+="--arch_frame resnet18dilated "
OPTS+="--img_pool maxpool "
OPTS+="--num_channels 16 "
# binary mask, BCE loss, weighted loss
OPTS+="--binary_mask 1 "
OPTS+="--loss bce "
#OPTS+="--output_activation sigmoid2 "
OPTS+="--weighted_loss 1 "
# logscale in frequency
OPTS+="--num_mix 2 "
OPTS+="--log_freq 1 "

# frames-related
OPTS+="--num_frames 3 "
OPTS+="--stride_frames 24 "
OPTS+="--frameRate 8 "

# audio-related
OPTS+="--audLen 65535 "
OPTS+="--audRate 11025 "

OPTS+="--mode eval "
OPTS+="--batch_size_per_gpu 20 "
OPTS+="--workers 32 "
OPTS+="--recurrent 1 "
OPTS+="--num_vis 40 "

python -u main_new.py $OPTS
