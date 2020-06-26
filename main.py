import glob
import json
import os
import subprocess
import sys

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from classify import classify_video
from mean import get_mean
from model import generate_model
from opts import parse_opts

if __name__ == "__main__":
    opt = parse_opts()
    opt.mean = get_mean()
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_size = 112
    opt.sample_duration = 5
    opt.n_classes = 400

    model = generate_model(opt)
    print('loading model {}'.format(opt.model))
    model_data = torch.load(opt.model)
    assert opt.arch == model_data['arch']
    model.load_state_dict(model_data['state_dict'])
    model.eval()
    if opt.verbose:
        print(model)

    class_names = []
    with open('class_names_list') as f:
        for row in f:
            class_names.append(row[:-1])

    ffmpeg_loglevel = 'quiet'
    if opt.verbose:
        ffmpeg_loglevel = 'info'

    if os.path.exists('tmp'):
        subprocess.call('rm -rf tmp', shell=True)

    outputs = []

    video_list = glob.glob(os.path.join(opt.video_dir, '*.mp4'))
    video_list += glob.glob(os.path.join(opt.video_dir, '*.avi'))
    video_list.sort()
    pbar = tqdm(video_list)

    with open(os.devnull, "w") as ffmpeg_log:
        for video in pbar:
            video_id = video.split("/")[-1].split(".")[0]
            pbar.set_description(video_id)
            subprocess.call('mkdir tmp', shell=True)
            subprocess.call('ffmpeg -i {} tmp/image_%05d.jpg'.format(video),
                            shell=True, stdout=ffmpeg_log, stderr=ffmpeg_log)
            try:
                result = classify_video('tmp', video, class_names, model, opt)


                segments = result["clips"]
                feat = np.zeros((len(segments), 2048))
                i = 0
                for segment in segments:
                    feat[i] = segment["features"]
                    i += 1
                np.save(opt.output_dir + '/' + video_id, feat)
            except Exception as err:
                print(video_id, err)

            subprocess.call('rm -rf tmp', shell=True)

        if os.path.exists('tmp'):
            subprocess.call('rm -rf tmp', shell=True)
