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
#Options

rendered_output = './tmp/rendered.mp4'
arch = 'resnet3d50'
frame_folder = None
start = 30
end = 60
num_segments = 16
videoPath = './tmp'
imgPath = './tmp/frames'


#Load video
def load_video(video_hash):
    yt = YouTube('https://youtube.com/embed/%s?start=%d&end=%d' % (video_hash, start, end))
    video = yt.streams.all()[0]
    name = video.download('/tmp')
    #   Load model
    model = models.load_model(arch)

    av_categories = pd.read_csv('CVS_Actions.csv', delimiter=';').values.tolist()
    trax = pd.read_csv('audioTracks_urls.csv')

    # Get dataset categories
    #categories = models.load_categories()

    # Load the video frame transform
    transform = models.load_transform()

    # Obtain video frames
    if frame_folder is not None:
        print('Loading frames in {}'.format(frame_folder))
        import glob
        # here make sure after sorting the frame paths have the correct temporal order
        frame_paths = sorted(glob.glob(os.path.join(frame_folder, '*.jpg')))
        print(frame_paths)
        frames = load_frames(frame_paths)
    else:
        print('Extracting frames using ffmpeg...')
        frames = extract_frames(name, num_segments)


    # Prepare input tensor
    if arch == 'resnet3d50':
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
    
    print('RESULT ON ' + name)
    y = float(av_categories[idx[0]][1])*125
    x = float(av_categories[idx[0]][2])*125

    trax = trax.assign(dist = lambda row: np.sqrt( (x - row.valence)**2 + (y - row.energy)**2 ) )
    print(trax['dist'].min())
    match = trax.loc[trax['dist']==trax['dist'].min(),['artist', 'track', 'preview_url']]
    song = 'valence: '+str(x)+ ' arousal: '+str(y)+' '+match.iloc[0,0]+' - '+ match.iloc[0,1]
    print(match.iloc[0,0], match.iloc[0,1])
    print(x, y,)
    for i in range(0, 5):
        print('{:.3f} -> {} ->{}'.format(probs[i], idx[i],av_categories[idx[i]]) )
        print('result   cutegories',av_categories[idx[i]][0], av_categories[idx[i]][1])
    
    r = requests.get(match.iloc[0,2], allow_redirects=True)
    open('./tmp/preview.mp3', 'wb').write(r.content)
    # Render output frames with prediction text.
    if rendered_output is not None:
        clip = VideoFileClip(name)
        audioclip = AudioFileClip('./tmp/preview.mp3')
        txt_clip = TextClip(song,fontsize=16,color='white')
        clip_final = clip.set_audio(audioclip)
        video = CompositeVideoClip([clip_final, txt_clip])
        video.set_duration(30).write_videofile(rendered_output)