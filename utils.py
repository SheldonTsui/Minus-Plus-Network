import os
import shutil

import numpy as np
import librosa
import cv2

import subprocess as sp
from threading import Timer
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io.wavfile as wavfile
from scipy.misc import imsave
from mir_eval.separation import bss_eval_sources
import pdb

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def warpgrid(bs, HO, WO, warp=True):
    # meshgrid
    x = np.linspace(-1, 1, WO)
    y = np.linspace(-1, 1, HO)
    xv, yv = np.meshgrid(x, y)
    grid = np.zeros((bs, HO, WO, 2))
    grid_x = xv
    if warp:
        grid_y = (np.power(21, (yv+1)/2) - 11) / 10
    else:
        grid_y = np.log(yv * 10 + 11) / np.log(21) * 2 - 1
    grid[:, :, :, 0] = grid_x
    grid[:, :, :, 1] = grid_y
    grid = grid.astype(np.float32)
    return grid


def makedirs(path, remove=False):
    if os.path.exists(path):
        if remove:
            shutil.rmtree(path)
            print('removed existing directory...')
        else:
            return
    os.makedirs(path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val*weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        val = np.asarray(val)
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        if self.val is None:
            return 0.
        else:
            return self.val.tolist()

    def average(self):
        if self.avg is None:
            return 0.
        else:
            return self.avg.tolist()

    def sum_value(self):
        if self.sum is None:
            return 0.
        else:
            return self.sum.tolist()

def recover_rgb(img):
    for t, m, s in zip(img,
                       [0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225]):
        t.mul_(s).add_(m)
    img = (img.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
    return img


def magnitude2heatmap(mag, log=True, scale=200.):
    if log:
        mag = np.log10(mag + 1.)
    mag *= scale
    mag[mag > 255] = 255
    mag = mag.astype(np.uint8)
    mag_color = cv2.applyColorMap(mag, cv2.COLORMAP_JET)
    mag_color = mag_color[:, :, ::-1]
    return mag_color


def istft_reconstruction(mag, phase, hop_length=256):
    spec = mag.astype(np.complex) * np.exp(1j*phase)
    wav = librosa.istft(spec, hop_length=hop_length)
    return np.clip(wav, -1., 1.)


class VideoWriter:
    """ Combine numpy frames into video using ffmpeg

    Arguments:
        filename: name of the output video
        fps: frame per second
        shape: shape of video frame

    Properties:
        add_frame(frame):
            add a frame to the video
        add_frames(frames):
            add multiple frames to the video
        release():
            release writing pipe

    """

    def __init__(self, filename, fps, shape):
        self.file = filename
        self.fps = fps
        self.shape = shape

        # video codec
        ext = filename.split('.')[-1]
        if ext == "mp4":
            self.vcodec = "h264"
        else:
            raise RuntimeError("Video codec not supoorted.")

        # video writing pipe
        cmd = [
            "ffmpeg",
            "-y",                                     # overwrite existing file
            "-f", "rawvideo",                         # file format
            "-s", "{}x{}".format(shape[1], shape[0]), # size of one frame
            "-pix_fmt", "rgb24",                      # 3 channels
            "-r", str(self.fps),                      # frames per second
            "-i", "-",                                # input comes from a pipe
            "-an",                                    # not to expect any audio
            "-vcodec", self.vcodec,                   # video codec
            "-pix_fmt", "yuv420p",                  # output video in yuv420p
            self.file]

        self.pipe = sp.Popen(cmd, stdin=sp.PIPE, stderr=sp.PIPE, bufsize=10**9)

    def release(self):
        self.pipe.stdin.close()

    def add_frame(self, frame):
        assert len(frame.shape) == 3
        assert frame.shape[0] == self.shape[0]
        assert frame.shape[1] == self.shape[1]
        try:
            self.pipe.stdin.write(frame.tostring())
        except:
            _, ffmpeg_error = self.pipe.communicate()
            print(ffmpeg_error)

    def add_frames(self, frames):
        for frame in frames:
            self.add_frame(frame)


def kill_proc(proc):
    proc.kill()
    print('Process running overtime! Killed.')


def run_proc_timeout(proc, timeout_sec):
    # kill_proc = lambda p: p.kill()
    timer = Timer(timeout_sec, kill_proc, [proc])
    try:
        timer.start()
        proc.communicate()
    finally:
        timer.cancel()


def combine_video_audio(src_video, src_audio, dst_video, verbose=False):
    try:
        cmd = ["ffmpeg", "-y",
               "-loglevel", "quiet",
               "-i", src_video,
               "-i", src_audio,
               "-c:v", "copy",
               "-c:a", "aac",
               "-strict", "experimental",
               dst_video]
        proc = sp.Popen(cmd)
        run_proc_timeout(proc, 10.)

        if verbose:
            print('Processed:{}'.format(dst_video))
    except Exception as e:
        print('Error:[{}] {}'.format(dst_video, e))


# save video to the disk using ffmpeg
def save_video(path, tensor, fps=25):
    assert tensor.ndim == 4, 'video should be in 4D numpy array'
    L, H, W, C = tensor.shape
    writer = VideoWriter(
        path,
        fps=fps,
        shape=[H, W])
    for t in range(L):
        writer.add_frame(tensor[t])
    writer.release()


def save_audio(path, audio_numpy, sr):
    librosa.output.write_wav(path, audio_numpy, sr)


# Visualize predictions
def output_visuals(vis_rows, batch_data, outputs, args):
    # fetch data and predictions
    mag_mix = batch_data['mag_mix']
    phase_mix = batch_data['phase_mix']
    frames = batch_data['frames']
    infos = batch_data['infos']

    pred_masks_ = outputs['pred_masks']
    # pred_masks_refine_ = outputs['pred_masks_refine_']
    gt_masks_ = outputs['gt_masks']
    mag_mix_ = outputs['mag_mix']
    weight_ = outputs['weight']

    # unwarp log scale
    N = args.num_mix
    B = mag_mix.size(0)
    pred_masks_linear = [None for n in range(N)]
    gt_masks_linear = [None for n in range(N)]
    for n in range(N):
        if args.log_freq:
            grid_unwarp = torch.from_numpy(
                warpgrid(B, args.stft_frame//2+1, gt_masks_[0].size(3), warp=False)).to(args.device)
            pred_masks_linear[n] = F.grid_sample(pred_masks_[n], grid_unwarp)
            gt_masks_linear[n] = F.grid_sample(gt_masks_[n], grid_unwarp)
        else:
            pred_masks_linear[n] = pred_masks_[n]
            gt_masks_linear[n] = gt_masks_[n]

    # convert into numpy
    mag_mix = mag_mix.numpy()
    mag_mix_ = mag_mix_.detach().cpu().numpy()
    phase_mix = phase_mix.numpy()
    weight_ = weight_.detach().cpu().numpy()
    for n in range(N):
        pred_masks_[n] = pred_masks_[n].detach().cpu().numpy()
        # pred_masks_refine_[n] = pred_masks_refine_[n].detach().cpu().numpy()
        pred_masks_linear[n] = pred_masks_linear[n].detach().cpu().numpy()
        gt_masks_[n] = gt_masks_[n].detach().cpu().numpy()
        gt_masks_linear[n] = gt_masks_linear[n].detach().cpu().numpy()

        # threshold if binary mask
        if args.binary_mask:
            pred_masks_[n] = (pred_masks_[n] > args.mask_thres).astype(np.float32)
            # pred_masks_refine_[n] = (pred_masks_refine_[n] > args.mask_thres).astype(np.float32)
            pred_masks_linear[n] = (pred_masks_linear[n] > args.mask_thres).astype(np.float32)

    # loop over each sample
    for j in range(B):
        row_elements = []

        # video names
        prefix = []
        for n in range(N):
            prefix.append('-'.join(infos[n][0][j].split('/')[-2:]).split('.')[0])
        prefix = '+'.join(prefix)
        makedirs(os.path.join(args.vis, prefix))

        # save mixture
        mix_wav = istft_reconstruction(mag_mix[j, 0], phase_mix[j, 0], hop_length=args.stft_hop)
        mix_amp = magnitude2heatmap(mag_mix_[j, 0])
        weight = magnitude2heatmap(weight_[j, 0], log=False, scale=100.)
        filename_mixwav = os.path.join(prefix, 'mix.wav')
        filename_mixmag = os.path.join(prefix, 'mix.jpg')
        filename_weight = os.path.join(prefix, 'weight.jpg')
        imsave(os.path.join(args.vis, filename_mixmag), mix_amp[::-1, :, :])
        imsave(os.path.join(args.vis, filename_weight), weight[::-1, :])
        wavfile.write(os.path.join(args.vis, filename_mixwav), args.audRate, mix_wav)
        row_elements += [{'text': prefix}, {'image': filename_mixmag, 'audio': filename_mixwav}]

        # save each component
        preds_wav = [None for n in range(N)]
        for n in range(N):
            # GT and predicted audio recovery
            gt_mag = mag_mix[j, 0] * gt_masks_linear[n][j, 0]
            gt_wav = istft_reconstruction(gt_mag, phase_mix[j, 0], hop_length=args.stft_hop)
            pred_mag = mag_mix[j, 0] * pred_masks_linear[n][j, 0]
            preds_wav[n] = istft_reconstruction(pred_mag, phase_mix[j, 0], hop_length=args.stft_hop)

            # output masks
            filename_gtmask = os.path.join(prefix, 'gtmask{}.jpg'.format(n+1))
            filename_predmask = os.path.join(prefix, 'predmask{}.jpg'.format(n+1))
            # filename_predmask_refine = os.path.join(prefix, 'predmaskref{}.jpg'.format(n+1))
            gt_mask = (np.clip(gt_masks_[n][j, 0], 0, 1) * 255).astype(np.uint8)
            pred_mask = (np.clip(pred_masks_[n][j, 0], 0, 1) * 255).astype(np.uint8)
            # pred_mask_refine = (np.clip(pred_masks_refine_[n][j, 0], 0, 1) * 255).astype(np.uint8)
            imsave(os.path.join(args.vis, filename_gtmask), gt_mask[::-1, :])
            imsave(os.path.join(args.vis, filename_predmask), pred_mask[::-1, :])
            # imsave(os.path.join(args.vis, filename_predmask_refine), pred_mask_refine[::-1, :])

            # ouput spectrogram (log of magnitude, show colormap)
            filename_gtmag = os.path.join(prefix, 'gtamp{}.jpg'.format(n+1))
            filename_predmag = os.path.join(prefix, 'predamp{}.jpg'.format(n+1))
            gt_mag = magnitude2heatmap(gt_mag)
            pred_mag = magnitude2heatmap(pred_mag)
            imsave(os.path.join(args.vis, filename_gtmag), gt_mag[::-1, :, :])
            imsave(os.path.join(args.vis, filename_predmag), pred_mag[::-1, :, :])

            # output audio
            filename_gtwav = os.path.join(prefix, 'gt{}.wav'.format(n+1))
            filename_predwav = os.path.join(prefix, 'pred{}.wav'.format(n+1))
            wavfile.write(os.path.join(args.vis, filename_gtwav), args.audRate, gt_wav)
            wavfile.write(os.path.join(args.vis, filename_predwav), args.audRate, preds_wav[n])

            # output video
            frames_tensor = [recover_rgb(frames[n][j, :, t]) for t in range(args.num_frames)]
            frames_tensor = np.asarray(frames_tensor)
            path_video = os.path.join(args.vis, prefix, 'video{}.mp4'.format(n+1))
            save_video(path_video, frames_tensor, fps=args.frameRate/args.stride_frames)

            # combine gt video and audio
            filename_av = os.path.join(prefix, 'av{}.mp4'.format(n+1))
            combine_video_audio(
                path_video,
                os.path.join(args.vis, filename_gtwav),
                os.path.join(args.vis, filename_av))

            row_elements += [
                {'video': filename_av},
                {'image': filename_predmag, 'audio': filename_predwav},
                {'image': filename_gtmag, 'audio': filename_gtwav},
                {'image': filename_predmask},
                {'image': filename_gtmask}]

        row_elements += [{'image': filename_weight}]
        vis_rows.append(row_elements)


# Calculate metrics
def calc_metrics(batch_data, outputs, args):
    # meters
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()

    # fetch data and predictions
    mag_mix = batch_data['mag_mix']
    phase_mix = batch_data['phase_mix']
    audios = batch_data['audios']

    pred_masks_ = outputs['pred_masks']

    # unwarp log scale
    N = args.num_mix
    B = mag_mix.size(0)
    pred_masks_linear = [None for n in range(N)]
    for n in range(N):
        if args.log_freq:
            grid_unwarp = torch.from_numpy(
                warpgrid(B, args.stft_frame//2+1, pred_masks_[0].size(3), warp=False)).to(args.device)
            pred_masks_linear[n] = F.grid_sample(pred_masks_[n], grid_unwarp)
        else:
            pred_masks_linear[n] = pred_masks_[n]

    # convert into numpy
    mag_mix = mag_mix.numpy()
    phase_mix = phase_mix.numpy()
    for n in range(N):
        pred_masks_linear[n] = pred_masks_linear[n].detach().cpu().numpy()

        # threshold if binary mask
        if args.binary_mask:
            pred_masks_linear[n] = (pred_masks_linear[n] > args.mask_thres).astype(np.float32)

    # loop over each sample
    valid_num = 0
    for j in range(B):
        # save mixture
        mix_wav = istft_reconstruction(mag_mix[j, 0], phase_mix[j, 0], hop_length=args.stft_hop)

        # save each component
        preds_wav = [None for n in range(N)]
        for n in range(N):
            # Predicted audio recovery
            pred_mag = mag_mix[j, 0] * pred_masks_linear[n][j, 0]
            preds_wav[n] = istft_reconstruction(pred_mag, phase_mix[j, 0], hop_length=args.stft_hop)

        # separation performance computes
        L = preds_wav[0].shape[0]
        gts_wav = [None for n in range(N)]
        valid = True
        for n in range(N):
            gts_wav[n] = audios[n][j, 0:L].numpy()
            valid *= np.sum(np.abs(gts_wav[n])) > 1e-5
            valid *= np.sum(np.abs(preds_wav[n])) > 1e-5
        if valid:
            valid_num += 1
            sdr, sir, sar, _ = bss_eval_sources(
                np.asarray(gts_wav),
                np.asarray(preds_wav),
                False)
            sdr_mix, _, _, _ = bss_eval_sources(
                np.asarray(gts_wav),
                np.asarray([mix_wav[0:L] for n in range(N)]),
                False)
            #print("sdr_m, sdr, sir, sar:", sdr_mix, sdr, sir, sar)
            sdr_mix_meter.update(sdr_mix.mean())
            sdr_meter.update(sdr.mean())
            sir_meter.update(sir.mean())
            sar_meter.update(sar.mean())

    return [sdr_mix_meter.sum_value(),
            sdr_meter.sum_value(),
            sir_meter.sum_value(),
            sar_meter.sum_value(),
            valid_num]

