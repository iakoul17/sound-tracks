"""Test pre-trained RGB model on a single video.

Date: 01/15/18
Authors: Bolei Zhou and Alex Andonian

This script accepts an mp4 video as the command line argument --video_file
and averages ResNet50 (trained on Moments) predictions on num_segment equally
spaced frames (extracted using ffmpeg).

Alternatively, one may instead provide the path to a directory containing
video frames saved as jpgs, which are sorted and forwarded through the model.

ResNet50 trained on Moments is used to predict the action for each frame,
and these class probabilities are average to produce a video-level predction.

Optionally, one can generate a new video --rendered_output from the frames
used to make the prediction with the predicted category in the top-left corner.

"""

import os
import argparse
import moviepy.editor as mpy
from moviepy.editor import *
from scipy.spatial import distance
import math
import numpy as np  
from pytube import YouTube #for youtube videos

import torch.optim
import torch.nn.parallel
from torch.nn import functional as F
import pandas as pd
import requests

import models

from utils import extract_frames, load_frames, render_frames


# options
parser = argparse.ArgumentParser(description="test TRN on a single video")
group = parser.add_mutually_exclusive_group(required=False)
group.add_argument('--video_file', type=str, default=None)
group.add_argument('--frame_folder', type=str, default=None)
parser.add_argument('--rendered_output', type=str, default='./tmp/rendered.mp4')
parser.add_argument('--num_segments', type=int, default=16)
parser.add_argument('--arch', type=str, default='resnet3d50', choices=['resnet50', 'resnet3d50'])
args = parser.parse_args()
#Load video
videoPath = './tmp'
imgPath = './tmp/frames'
os.makedirs(videoPath, exist_ok=True)
os.makedirs(imgPath, exist_ok=True)
start = 30
end = 60
video_hash = 'gu58bxKMfhQ'
yt = YouTube('https://youtube.com/embed/%s?start=%d&end=%d' % (video_hash, start, end))
video = yt.streams.all()[0]
name = video.download('/tmp')
# Load model
model = models.load_model(args.arch)

av_categories = pd.read_csv('CVS_Actions.csv', delimiter=';').values.tolist()
trax = pd.read_csv('audioTracks_urls.csv')

# Get dataset categories
#categories = models.load_categories()

# Load the video frame transform
transform = models.load_transform()

# Obtain video frames
if args.frame_folder is not None:
    print('Loading frames in {}'.format(args.frame_folder))
    import glob
    # here make sure after sorting the frame paths have the correct temporal order
    frame_paths = sorted(glob.glob(os.path.join(args.frame_folder, '*.jpg')))
    print(frame_paths)
    frames = load_frames(frame_paths)
else:
    print('Extracting frames using ffmpeg...')
    frames = extract_frames(name, args.num_segments)


# Prepare input tensor
if args.arch == 'resnet3d50':
    # [1, num_frames, 3, 224, 224]
    input = torch.stack([transform(frame) for frame in frames], 1).unsqueeze(0)
else:
    # [num_frames, 3, 224, 224]
    input = torch.stack([transform(frame) for frame in frames])

# Make video prediction
with torch.no_grad():
    logits = model(input)
    h_x = F.softmax(logits, 1).mean(dim=0)
    probs, idx = h_x.sort(0, True)

# Output the prediction.
video_name = args.frame_folder if args.frame_folder is not None else args.video_file
print('RESULT ON ' + name)
y = float(av_categories[idx[0]][1])*125
x = float(av_categories[idx[0]][2])*125

trax = trax.assign(dist = lambda row: np.sqrt( (x - row.valence)**2 + (y - row.energy)**2 ) )
print(trax['dist'].min())
match = trax.loc[trax['dist']==trax['dist'].min(),['artist', 'track', 'preview_url']]

print(match.iloc[0,0], match.iloc[0,1])
print(x,type(x), y,type(y))
for i in range(0, 5):
    print('{:.3f} -> {} ->{}'.format(probs[i], idx[i],av_categories[idx[i]]) )
    print('result   cutegories',av_categories[idx[i]][0], av_categories[idx[i]][1])

r = requests.get(match.iloc[0,2], allow_redirects=True)
open('./tmp/preview.mp3', 'wb').write(r.content)
# Render output frames with prediction text.
if args.rendered_output is not None:
    
    
    clip = VideoFileClip(name)
    audioclip = AudioFileClip('./tmp/preview.mp3')
    clip_final = clip.set_audio(audioclip)
    clip_final.set_duration(30).write_videofile(args.rendered_output)
