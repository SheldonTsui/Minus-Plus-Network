#!/bin/bash 
OPTS=""
OPTS+="--id MUSIC "
OPTS+="--list_train ./data/train_wav.csv "
OPTS+="--list_val ./data/val_wav.csv "
OPTS+="--dup_trainset 100 "
OPTS+="--margin 10 "

# Models
OPTS+="--arch_sound unet7 "
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
#OPTS+="--mask_thres 0.33 "
OPTS+="--log_freq 1 "

# frames-related
OPTS+="--num_frames 3 "
OPTS+="--stride_frames 24 "
OPTS+="--frameRate 8 "

# audio-related
OPTS+="--audLen 95800 "
OPTS+="--audRate 16000 "
OPTS+="--stft_frame 1500 "
OPTS+="--stft_hop 375 "

# learning params
OPTS+="--num_gpus 4 "
OPTS+="--workers 48 "
OPTS+="--batch_size_per_gpu 20 "
OPTS+="--lr_frame 1e-4 "
OPTS+="--lr_sound 1e-3 "
OPTS+="--num_epoch 100 "
OPTS+="--lr_steps 40 80 "

# display, viz
OPTS+="--disp_iter 20 "
OPTS+="--num_vis 40 "
OPTS+="--num_val 400 "

OPTS+="--mode train "
OPTS+="--forward_mode Minus "
OPTS+="--init_clip False "
OPTS+="--exp_name Minus_only "
OPTS+="--gpu_ids 0,1,2,3 "
OPTS+="--need_loss_ratio True "

python -u main.py $OPTS
