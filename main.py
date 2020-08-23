# System libs
import os
import random
import time

# Numerical libs
import torch
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter

# Our libs
from arguments import ArgParser
from dataset import MUSICMixDataset
from models import ModelBuilder, activate
from utils import AverageMeter, warpgrid, makedirs, calc_metrics, output_visuals
from viz import plot_loss_metrics, HTMLVisualizer
import json
import pdb


# Network wrapper, defines forward pass
class NetWrapper(torch.nn.Module):
    def __init__(self, nets, crit, mode='Minus'):
        super(NetWrapper, self).__init__()
        self.net_sound_M, self.net_frame_M, self.net_sound_P = nets
        self.crit = crit
        self.mode = mode

    @staticmethod
    def choose_max(index_record, total_energy):
        MIN = -100
        for index in index_record:
            Len = len(index)
            total_energy[list(range(Len)), index] = np.ones(Len) * MIN

        max_index = np.argmax(total_energy, axis=1)

        return max_index
    
    @staticmethod
    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def forward(self, batch_data, args):
        ### prepare data
        mag_mix = batch_data['mag_mix']
        mags = batch_data['mags']
        frames = batch_data['frames']
        mag_mix = mag_mix + 1e-10
        mag_mix_tmp = mag_mix.clone()

        N = args.num_mix
        B = mag_mix.size(0)
        T = mag_mix.size(3)

        # 0.0 warp the spectrogram
        if args.log_freq:
            grid_warp = torch.from_numpy(
                warpgrid(B, 256, T, warp=True)).to(args.device)
            mag_mix = F.grid_sample(mag_mix, grid_warp)
            for n in range(N):
                mags[n] = F.grid_sample(mags[n], grid_warp)

        # 0.1 calculate loss weighting coefficient: magnitude of input mixture
        if args.weighted_loss:
            weight = torch.log1p(mag_mix)
            weight = torch.clamp(weight, 1e-3, 10)
        else:
            weight = torch.ones_like(mag_mix)

        # 0.2 ground truth masks are computed after warpping!
        # Please notice that, gt_masks are unordered
        gt_masks = [None for n in range(N)]
        for n in range(N):
            if args.binary_mask:
                # for simplicity, mag_N > 0.5 * mag_mix
                gt_masks[n] = (mags[n] > 0.5 * mag_mix).float()
            else:
                gt_masks[n] = mags[n] / mag_mix
                # clamp to avoid large numbers in ratio masks
                gt_masks[n].clamp_(0., 2.)

        ### Minus part
        if 'Minus' not in self.mode:
            self.requires_grad(self.net_sound_M, False)
            self.requires_grad(self.net_frame_M, False)

        feat_frames = [None for n in range(N)]
        ordered_pred_masks = [None for n in range(N)]
        ordered_pred_mags = [None for n in range(N)]

        # Step1: obtain all the frame features
        # forward net_frame_M -> Bx1xC
        for n in range(N):
            log_mag_mix = torch.log(mag_mix)
            feat_frames[n] = self.net_frame_M.forward_multiframe(frames[n])
            feat_frames[n] = activate(feat_frames[n], args.img_activation)
        
        # Step2: separate the sounds one by one
        # forward net_sound_M -> BxCxHxW
        if args.log_freq:
            grid_unwarp = torch.from_numpy(
                warpgrid(B, args.stft_frame//2+1, 256, warp=False)).to(args.device)
        index_record = []
        for n in range(N):
            log_mag_mix = torch.log(mag_mix).detach()
            feat_sound = self.net_sound_M(log_mag_mix)
            _, C, H, W = feat_sound.shape
            feat_sound = feat_sound.view(B, C, -1)            
 
            # obtain current separated sound
            energy_list = []
            tmp_masks = []
            tmp_pred_mags = []

            for feat_frame in feat_frames:
                cur_pred_mask = torch.bmm(feat_frame.unsqueeze(1), feat_sound).view(B, 1, H, W)
                cur_pred_mask = activate(cur_pred_mask, args.output_activation)
                tmp_masks.append(cur_pred_mask)
                # Here we cut off the loss flow from Minus net to Plus net
                # in order to train more steadily
                if args.log_freq:
                    cur_pred_mask_unwrap = F.grid_sample(cur_pred_mask.detach(), grid_unwarp)
                    if args.binary_mask:
                        cur_pred_mask_unwrap = (cur_pred_mask_unwrap > args.mask_thres).float()
                else:
                    cur_pred_mask_unwrap = cur_pred_mask.detach()
                cur_pred_mag = cur_pred_mask_unwrap * mag_mix_tmp
                tmp_pred_mags.append(cur_pred_mag)
                energy_list.append(np.array(cur_pred_mag.view(B, -1).mean(dim=1).cpu().data))
            
            total_energy = np.stack(energy_list, axis=1)
            # _, cur_index = torch.max(total_energy)      
            cur_index = self.choose_max(index_record, total_energy)
            index_record.append(cur_index)

            masks = torch.stack(tmp_masks, dim=0)
            ordered_pred_masks[n] = masks[cur_index, list(range(B))]
            pred_mags = torch.stack(tmp_pred_mags, dim=0)
            ordered_pred_mags[n] = pred_mags[cur_index, list(range(B))]

            #log_mag_mix = log_mag_mix - log_mag_mix * pred_masks[n]
            mag_mix = mag_mix - mag_mix * ordered_pred_masks[n] + 1e-10
        
        # just for swap pred_masks, in order to compute loss conveniently
        # since gt_masks are unordered, we must transfer ordered_pred_masks to unordered
        index_record = np.stack(index_record, axis=1)
        total_masks = torch.stack(ordered_pred_masks, dim=1)
        total_pred_mags = torch.stack(ordered_pred_mags, dim=1)
        unordered_pred_masks = []
        unordered_pred_mags = []
        for n in range(N):
            mask_index = np.where(index_record == n)
            if args.binary_mask:
                unordered_pred_masks.append(total_masks[mask_index])
                unordered_pred_mags.append(total_pred_mags[mask_index])
            else:
                unordered_pred_masks.append(total_masks[mask_index] * 2)
                unordered_pred_mags.append(total_pred_mags[mask_index])
        
        ### Plus part
        if 'Plus' in self.mode:
            pre_sum = torch.zeros_like(unordered_pred_masks[0]).to(args.device)
            Plus_pred_masks = []
            for n in range(N):
                unordered_pred_mag = unordered_pred_mags[n].log()
                unordered_pred_mag = F.grid_sample(unordered_pred_mag, grid_warp)
                input_concat = torch.cat((pre_sum, unordered_pred_mag), dim=1)
                
                residual_mask = activate(self.net_sound_P(input_concat), args.sound_activation)
                Plus_pred_masks.append(unordered_pred_masks[n] + residual_mask)
                
                pre_sum = pre_sum.sum(dim=1, keepdim=True).detach()
                
            unordered_pred_masks = Plus_pred_masks
            
        # loss
        if args.need_loss_ratio:
            err = 0
            for n in range(N):
                err += self.crit(unordered_pred_masks[n], gt_masks[n], weight) / N * 2 ** (n-1)
        else:
            err = self.crit(unordered_pred_masks, gt_masks, weight).reshape(1)

        if 'Minus' in self.mode:
            res_mag_mix = torch.exp(log_mag_mix)
            err_remain = torch.mean(weight * torch.clamp(res_mag_mix - 1e-2, min=0))
            err += err_remain

        outputs = {'pred_masks': unordered_pred_masks, 'gt_masks': gt_masks,
            'mag_mix': mag_mix, 'mags': mags, 'weight': weight}

        return err, outputs

class MP_Trainer(torch.nn.Module):
    def __init__(self, netwrapper, optimizer, args):
        super().__init__()

        self.mode = args.mode
        self.netwrapper = netwrapper
        self.optimizer = optimizer
        self.args = args

        self.history = {
            'train': {'epoch': [], 'err': []},
            'val': {'epoch': [], 'err': [], 'sdr': [], 'sir': [], 'sar': []}
        }

        if self.mode == 'train':
            self.writer = SummaryWriter(log_dir=os.path.join('./logs', self.args.exp_name))
        
        self.epoch = 0  # epoch initialize

    def evaluate(self, loader):
        print('Evaluating at {} epochs...'.format(self.epoch))
        torch.set_grad_enabled(False)

        # remove previous viz results
        makedirs(self.args.vis, remove=True)

        self.netwrapper.eval()

        # initialize meters
        loss_meter = AverageMeter()
        sdr_mix_meter = AverageMeter()
        sdr_meter = AverageMeter()
        sir_meter = AverageMeter()
        sar_meter = AverageMeter()

        # initialize HTML header
        visualizer = HTMLVisualizer(os.path.join(self.args.vis, 'index.html'))
        header = ['Filename', 'Input Mixed Audio']
        for n in range(1, self.args.num_mix+1):
            header += ['Video {:d}'.format(n),
                    'Predicted Audio {:d}'.format(n),
                    'GroundTruth Audio {}'.format(n),
                    'Predicted Mask {}'.format(n),
                    'GroundTruth Mask {}'.format(n)]
        header += ['Loss weighting']
        visualizer.add_header(header)
        vis_rows = []
        eval_num = 0
        valid_num = 0

        #for i, batch_data in enumerate(self.loader['eval']):
        for i, batch_data in enumerate(loader):
            # forward pass
            eval_num += batch_data['mag_mix'].shape[0]
            with torch.no_grad():
                err, outputs = self.netwrapper.forward(batch_data, args)
                err = err.mean()

            if self.mode == 'train':
                self.writer.add_scalar('data/val_loss', err, self.args.epoch_iters * self.epoch + i)

            loss_meter.update(err.item())
            print('[Eval] iter {}, loss: {:.4f}'.format(i, err.item()))

            # calculate metrics
            sdr_mix, sdr, sir, sar, cur_valid_num = calc_metrics(batch_data, outputs, self.args)
            print("sdr_mix, sdr, sir, sar: ", sdr_mix, sdr, sir, sar)
            sdr_mix_meter.update(sdr_mix)
            sdr_meter.update(sdr)
            sir_meter.update(sir)
            sar_meter.update(sar)
            valid_num += cur_valid_num
            '''
            # output visualization
            if len(vis_rows) < self.args.num_vis:
                output_visuals(vis_rows, batch_data, outputs, self.args)
            '''
        metric_output = '[Eval Summary] Epoch: {}, Loss: {:.4f}, ' \
            'SDR_mixture: {:.4f}, SDR: {:.4f}, SIR: {:.4f}, SAR: {:.4f}'.format(
                self.epoch, loss_meter.average(),
                sdr_mix_meter.sum_value()/eval_num,
                sdr_meter.sum_value()/eval_num,
                sir_meter.sum_value()/eval_num,
                sar_meter.sum_value()/eval_num
        )
        if valid_num / eval_num < 0.8:
            metric_output += ' ---- Invalid ---- '
        
        print(metric_output)
        learning_rate = ' lr_sound: {}, lr_frame: {}'.format(self.args.lr_sound, self.args.lr_frame)
        with open(self.args.log, 'a') as F:
            F.write(metric_output + learning_rate + '\n')
        
        self.history['val']['epoch'].append(self.epoch)
        self.history['val']['err'].append(loss_meter.average())
        self.history['val']['sdr'].append(sdr_meter.sum_value()/eval_num)
        self.history['val']['sir'].append(sir_meter.sum_value()/eval_num)
        self.history['val']['sar'].append(sar_meter.sum_value()/eval_num)
        '''
        print('Plotting html for visualization...')
        visualizer.add_rows(vis_rows)
        visualizer.write_html()
        '''
        # Plot figure
        if self.epoch > 0:
            print('Plotting figures...')
            plot_loss_metrics(self.args.ckpt, self.history)


    def train(self, loader):
        torch.set_grad_enabled(True)
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # switch to train mode
        self.netwrapper.train()
        
        # main loop
        torch.cuda.synchronize()
        tic = time.perf_counter()
        for i, batch_data in enumerate(loader):
            # measure data time
            torch.cuda.synchronize()
            data_time.update(time.perf_counter() - tic)

            # forward pass
            self.netwrapper.zero_grad()
            err, outputs = self.netwrapper.forward(batch_data, args)
            err = err.mean()

            self.writer.add_scalar('data/loss', err.mean(), self.args.epoch_iters * self.epoch + i)

            # backward
            err.backward()
            self.optimizer.step()

            # measure total time
            torch.cuda.synchronize()
            batch_time.update(time.perf_counter() - tic)
            tic = time.perf_counter()

            # display
            if i % self.args.disp_iter == 0:
                print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                    'lr_sound: {}, lr_frame: {}, '
                    'loss: {:.4f}'
                    .format(self.epoch, i, self.args.epoch_iters,
                            batch_time.average(), data_time.average(),
                            self.args.lr_sound, self.args.lr_frame,
                            err.item()))
                fractional_epoch = self.epoch - 1 + 1. * i / self.args.epoch_iters
                self.history['train']['epoch'].append(fractional_epoch)
                self.history['train']['err'].append(err.item())


    def checkpoint(self):

        print('Saving checkpoints at {} epochs.'.format(self.epoch))
        torch.save(self.history,
                '{}/history_{:03d}.pth'.format(self.args.ckpt, self.epoch))
        
        torch.save(self.netwrapper.module.net_sound_M.state_dict(),
                '{}/sound_M_{:03d}.pth'.format(self.args.ckpt, self.epoch))
        torch.save(self.netwrapper.module.net_frame_M.state_dict(),
                '{}/frame_M_{:03d}.pth'.format(self.args.ckpt, self.epoch))
        torch.save(self.netwrapper.module.net_sound_P.state_dict(),
                '{}/sound_P_{:03d}.pth'.format(self.args.ckpt, self.epoch))


    @staticmethod
    def create_optimizer(nets, args):
        (net_sound_M, net_frame_M, net_sound_P) = nets
        param_groups = [{'params': net_sound_M.parameters(), 'lr': args.lr_sound},
                        {'params': net_sound_P.parameters(), 'lr': args.lr_sound},
                        {'params': net_frame_M.features.parameters(), 'lr': args.lr_frame},
                        {'params': net_frame_M.fc.parameters(), 'lr': args.lr_sound}]
        return torch.optim.SGD(param_groups, momentum=args.beta1, weight_decay=args.weight_decay)


    def adjust_learning_rate(self):
        self.args.lr_sound *= 0.1
        self.args.lr_frame *= 0.1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.1


def main(args):
    # Network Builders
    builder = ModelBuilder()
    net_sound_M = builder.build_sound(
        arch=args.arch_sound,
        fc_dim=args.num_channels,
        weights=args.weights_sound_M)
    net_frame_M = builder.build_frame(
        arch=args.arch_frame,
        fc_dim=args.num_channels,
        pool_type=args.img_pool,
        weights=args.weights_frame_M)
    
    net_sound_P = builder.build_sound(
        input_nc=2,
        arch=args.arch_sound,
        # fc_dim=args.num_channels,
        fc_dim=1,
        weights=args.weights_sound_P)
    
    nets = (net_sound_M, net_frame_M, net_sound_P)
    crit = builder.build_criterion(arch=args.loss)

    # Wrap networks
    # set netwrapper forward mode
    # there are there modes for different training stages
    # ['Minus', 'Plus', 'Minus_Plus']
    netwrapper = NetWrapper(nets, crit, mode=args.forward_mode)
    netwrapper = torch.nn.DataParallel(netwrapper, device_ids=range(args.num_gpus))
    netwrapper.to(args.device)

    # Dataset and Loader
    dataset_train = MUSICMixDataset(
        args.list_train, args, split='train')
    dataset_val = MUSICMixDataset(
        args.list_val, args, max_sample=args.num_val, split='val')

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=True)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False)
    args.epoch_iters = len(dataset_train) // args.batch_size
    print('1 Epoch = {} iters'.format(args.epoch_iters))
    
    # Set up optimizer
    optimizer = MP_Trainer.create_optimizer(nets, args)

    mp_trainer = MP_Trainer(netwrapper, optimizer, args)

    # Eval firstly
    mp_trainer.evaluate(loader_val)
    if mp_trainer.mode == 'eval':
        print('Evaluation Done!')
    else:
        # start training
        for epoch in range(1, args.num_epoch + 1):
            mp_trainer.epoch = epoch
            mp_trainer.train(loader_train)

            # Evaluation and visualization
            if epoch % args.eval_epoch == 0:
                mp_trainer.evaluate(loader_val)

                # checkpointing
                mp_trainer.checkpoint()

            # adjust learning rate
            if epoch in args.lr_steps:
                mp_trainer.adjust_learning_rate()

        print('Training Done!')
        mp_trainer.writer.close()


if __name__ == '__main__':
    # arguments
    parser = ArgParser()
    args = parser.parse_train_arguments()
    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.device = torch.device("cuda")

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids 

    # experiment name
    if args.mode == 'train':
        args.id += '-{}mix'.format(args.num_mix)
        if args.log_freq:
            args.id += '-LogFreq'
        args.id += '-{}-{}'.format(args.arch_frame, args.arch_sound)
        args.id += '-frames{}stride{}'.format(args.num_frames, args.stride_frames)
        args.id += '-{}'.format(args.img_pool)
        if args.binary_mask:
            assert args.loss == 'bce', 'Binary Mask should go with BCE loss'
            args.id += '-binary'
        else:
            args.id += '-ratio'
        if args.weighted_loss:
            args.id += '-weightedLoss'
        args.id += '-channels{}'.format(args.num_channels)
        args.id += '-epoch{}'.format(args.num_epoch)
        args.id += '-step' + '_'.join([str(x) for x in args.lr_steps])
        args.id += '_' + args.exp_name

    print('Model ID: {}'.format(args.id))

    # paths to save/load output
    args.ckpt = os.path.join(args.ckpt, args.id)
    args.vis = os.path.join(args.ckpt, 'visualization/')
    args.log = os.path.join(args.ckpt, 'running_log.txt')
    pretrained_path = ''
    if args.mode == 'train':
        makedirs(args.ckpt, remove=True)
        args_path = os.path.join(args.ckpt, 'args.json')
        args_store = vars(args).copy()
        args_store['device'] = None
        with open(args_path, 'w') as json_file:
            json.dump(args_store, json_file)
        
    elif args.mode == 'eval':
        args.weights_sound_M = os.path.join(args.ckpt, 'sound_M_best.pth')
        args.weights_frame_M = os.path.join(args.ckpt, 'frame_M_best.pth')
        args.weights_sound_P = os.path.join(args.ckpt, 'sound_P_best.pth')

    # initialize best error with a big number
    args.best_err = float("inf")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
