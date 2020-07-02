import tensorflow as tf
from hparam import hparam as hp
from tensorpack.predict.base import OfflinePredictor
from tensorpack.predict.config import PredictConfig
from tensorpack.tfutils.sessinit import SaverRestore
from tensorpack.tfutils.sessinit import ChainInit
from models import Net2
from data_load import get_mfccs_and_spectrogram_directly
from convert import get_eval_input_names, get_eval_output_names, convert
from audio import  spec2wav
import os
import numpy as np
import moviepy.editor

if __name__ == '__main__':
    model = Net2()

    hp.set_hparam_yaml('hparams/default.yaml')
    
    logdir = '/data/private/vc/logdir'
    logdir1 = os.path.join(logdir, 'kek/train1')
    logdir2 = os.path.join(logdir, 'musk/train2')
    ckpt1 = tf.train.latest_checkpoint(logdir1)
    ckpt2 = tf.train.latest_checkpoint(logdir2)
    session_inits = []
    if ckpt2:
        session_inits.append(SaverRestore(ckpt2))
    if ckpt1:
        session_inits.append(SaverRestore(ckpt1, ignore=['global_step']))
    pred_conf = PredictConfig(
        model=model,
        input_names=get_eval_input_names(),
        output_names=get_eval_output_names(),
        session_init=ChainInit(session_inits))
    predictor = OfflinePredictor(pred_conf)
    #data_dir = '/data/private/vc/datasets/musk_speeches/musk_speech_1001'
    data_dir = '/data/private/vc/datasets/test_voice'
    input_audio = os.path.join(data_dir, os.listdir(data_dir)[-1])
    
    input_audio_clip = moviepy.editor.AudioFileClip(input_audio, fps=hp.default.sr)
    cur = 0
    duration = input_audio_clip.duration
    
    audio_clips = []
    y_audio_clips = []
    while cur < duration:
        print(f'cur = {cur}, right = {min(cur + hp.default.duration, duration)}')
        subclip = input_audio_clip.subclip(cur, min(cur + hp.default.duration, duration))
        print(subclip.to_soundarray())
        pr, y_s, pp = get_mfccs_and_spectrogram_directly(subclip.to_soundarray()[:,0])
        pred_spec, y_spec, ppgs = predictor(pr.reshape(1, *pr.shape), y_s.reshape(1, *y_s.shape), pp.reshape(1, *pp.shape))
        audio, y_audio, ppgs = convert(pred_spec, y_spec, ppgs)

        audio_clip_1 = moviepy.audio.AudioClip.AudioArrayClip(audio.reshape(-1).repeat(2).reshape(-1, 2), hp.default.sr)
        audio_clips.append(audio_clip_1)
        y_audio_clip_1 = moviepy.audio.AudioClip.AudioArrayClip(y_audio.reshape(-1).repeat(2).reshape(-1, 2), hp.default.sr)
        y_audio_clips.append(y_audio_clip_1)

        cur += hp.default.duration


    audio_clip = moviepy.editor.concatenate_audioclips(audio_clips)
    audio_clip.write_audiofile('audio.wav')
    y_audio_clip = moviepy.editor.concatenate_audioclips(y_audio_clips)
    y_audio_clip.write_audiofile('y_audio.wav')

