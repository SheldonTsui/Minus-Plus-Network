import os
import random
from .base import BaseDataset
import numpy as np
import sys
import pdb

def normalize(audio_data, re_factor=0.8):
    EPS = 1e-3
    min_data = audio_data.min()
    audio_data -= min_data
    max_data = audio_data.max()
    audio_data /= max_data + EPS 
    audio_data -= 0.5 
    audio_data *= 2
    audio_data *= re_factor

    return audio_data.clip(-1, 1)

def swap(input, order):
    len_order = len(order)
    if len_order < 2:
        return input
    elif len_order == 2:
        return input[order[0]], input[order[1]]
    elif len_order == 3:
        return input[order[0]], input[order[1]], input[order[2]]
    elif len_order == 4:
        return input[order[0]], input[order[1]], input[order[2]], input[order[3]]
    else:
        print("error")

class MUSICMixDataset(BaseDataset):
    def __init__(self, list_sample, opt, **kwargs):
        super(MUSICMixDataset, self).__init__(
            list_sample, opt, **kwargs)
        self.fps = opt.frameRate
        self.num_mix = opt.num_mix
        self.ade_path = opt.ade_path
        self.margin = opt.margin

    def __getitem__(self, index):
        N = self.num_mix
        frames = [None for n in range(N)]
        audios = [None for n in range(N)]
        infos = [[] for n in range(N)]
        path_frames = [[] for n in range(N)]
        path_audios = ['' for n in range(N)]
        center_frames = [0 for n in range(N)]

        # the first video
        infos[0] = self.list_sample[index]

        # sample other videos
        if not self.split == 'train':
            random.seed(index)
        for n in range(1, N):
            loop_flag = True
            while loop_flag:
                indexN = random.randint(0, len(self.list_sample)-1)
                if indexN != index:
                    break
            #indexN = random.randint(0, len(self.list_sample)-1)
            infos[n] = self.list_sample[indexN]

        # select frames
        idx_margin = max(
            int(self.fps * self.margin), (self.num_frames // 2) * self.stride_frames)
        for n, infoN in enumerate(infos):
            path_audioN, path_frameN, count_framesN = infoN

            if self.split == 'train':
                # random, not to sample start and end n-frames
                center_frameN = random.randint(
                    idx_margin+1, int(count_framesN)-idx_margin)
            else:
                center_frameN = int(count_framesN) // 2
            center_frames[n] = center_frameN

            # absolute frame/audio paths
            for i in range(self.num_frames):
                idx_offset = (i - self.num_frames // 2) * self.stride_frames
                path_frames[n].append(
                    os.path.join(
                        path_frameN,
                        '{:06d}.jpg'.format(center_frameN + idx_offset)))
            path_audios[n] = path_audioN

            if n > 0 and self.split == 'train':
                rand_number = random.randint(0, 19)
                if rand_number == 5:
                    path_audios[n] = 'special_silent'
                    rand_idx = random.randint(0, 49)
                    path_frames[n] = []
                    for _ in range(self.num_frames):
                        path_frames[n].append(os.path.join(self.ade_path, 'ADE_{:02d}.jpg'.format(rand_idx)))

        # load frames and audios, STFT
        try:
            for n, infoN in enumerate(infos):
                frames[n] = self._load_frames(path_frames[n])
                # jitter audio
                # center_timeN = (center_frames[n] - random.random()) / self.fps
                center_timeN = (center_frames[n] - 0.5) / self.fps
                audios[n] = self._load_audio(path_audios[n], center_timeN)
                audios[n] = normalize(audios[n])
            mag_mix, mags, phase_mix, order = self._mix_n_and_stft(audios)

        except Exception as e:
            print('Failed loading frame/audio: {}'.format(e))
            sys.exit(-1)
            # create dummy data
            mag_mix, mags, frames, audios, phase_mix = \
                self.dummy_mix_data(N)

        if self.split == 'train':
            #mags = swap(mags, order)
            #frames = swap(frames, order)
            ret_dict = {'mag_mix': mag_mix, 'frames': frames, 'mags': mags}
        else:
            ''' 
            mags = swap(mags, order)
            frames = swap(frames, order)
            audios = swap(audios, order)
            infos = swap(infos, order)
            ''' 
            ret_dict = {'mag_mix': mag_mix, 'frames': frames, 'mags': mags, 'audios': audios, \
                'phase_mix': phase_mix, 'infos': infos, 'order': order}

        return ret_dict
