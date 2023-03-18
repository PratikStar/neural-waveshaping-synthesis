from functools import partial
from typing import Callable, Optional, Sequence, Union

import gin
import librosa
import numpy as np
import torch
import torchcrepe
import resampy

from .upsampling import linear_interpolation
from ...utils import apply
from pathlib import Path
import scipy.io.wavfile as wavfile


CREPE_WINDOW_LENGTH = 1024
di_f0_estimates = {}

def convert_to_float32_audio_dupli(audio: np.ndarray):
    if audio.dtype == np.float32:
        return audio

    max_sample_value = np.iinfo(audio.dtype).max
    floating_point_audio = audio / max_sample_value
    return floating_point_audio.astype(np.float32)


def make_monophonic_dupli(audio: np.ndarray, strategy: str = "keep_left"):
    # deal with non stereo array formats
    if len(audio.shape) == 1:
        return audio
    elif len(audio.shape) != 2:
        raise ValueError("Unknown audio array format.")

    # deal with single audio channel
    if audio.shape[0] == 1:
        return audio[0]
    elif audio.shape[1] == 1:
        return audio[:, 0]
    # deal with more than two channels
    elif audio.shape[0] != 2 and audio.shape[1] != 2:
        raise ValueError("Expected stereo input audio but got too many channels.")

    # put channel first
    if audio.shape[1] == 2:
        audio = audio.T

    # make stereo audio monophonic
    if strategy == "keep_left":
        return audio[0]
    elif strategy == "keep_right":
        return audio[1]
    elif strategy == "sum":
        return np.mean(audio, axis=0)
    elif strategy == "diff":
        return audio[0] - audio[1]


def normalise_signal_dupli(audio: np.ndarray, factor: float):
    return audio / factor


def resample_audio_dupli(audio: np.ndarray, original_sr: float, target_sr: float):
    return resampy.resample(audio, original_sr, target_sr)

@gin.configurable
def extract_f0_with_crepe(
        audio: np.ndarray,
        sample_rate: float,
        hop_length: int = 128,
        minimum_frequency: float = 50.0,
        maximum_frequency: float = 2000.0,
        full_model: bool = True,
        batch_size: int = 2048,
        device: Union[str, torch.device] = "cpu",
        interpolate_fn: Optional[Callable] = linear_interpolation,
):
    # convert to torch tensor with channel dimension (necessary for CREPE)
    audio = torch.tensor(audio).unsqueeze(0)
    f0, confidence = torchcrepe.predict(
        audio,
        sample_rate,
        hop_length,
        minimum_frequency,
        maximum_frequency,
        "full" if full_model else "tiny",
        batch_size=batch_size,
        device=device,
        decoder=torchcrepe.decode.viterbi,
        # decoder=torchcrepe.decode.weighted_argmax,
        return_harmonicity=True,
    )

    f0, confidence = f0.squeeze().numpy(), confidence.squeeze().numpy()

    if interpolate_fn:
        f0 = interpolate_fn(
            f0, CREPE_WINDOW_LENGTH, hop_length, original_length=audio.shape[-1]
        )
        confidence = interpolate_fn(
            confidence,
            CREPE_WINDOW_LENGTH,
            hop_length,
            original_length=audio.shape[-1],
        )

    return f0, confidence


@gin.configurable
def extract_f0_with_pyin(
    audio: np.ndarray,
    sample_rate: float,
    minimum_frequency: float = 65.0,  # recommended minimum freq from librosa docs
    maximum_frequency: float = 2093.0,  # recommended maximum freq from librosa docs
    frame_length: int = 1024,
    hop_length: int = 128,
    fill_na: Optional[float] = None,
    interpolate_fn: Optional[Callable] = linear_interpolation,
):
    f0, _, voiced_prob = librosa.pyin(
        audio,
        sr=sample_rate,
        fmin=minimum_frequency,
        fmax=maximum_frequency,
        frame_length=frame_length,
        hop_length=hop_length,
        fill_na=fill_na,
    )

    if interpolate_fn:
        f0 = interpolate_fn(
            f0, frame_length, hop_length, original_length=audio.shape[-1]
        )
        voiced_prob = interpolate_fn(
            voiced_prob,
            frame_length,
            hop_length,
            original_length=audio.shape[-1],
        )

    return f0, voiced_prob
