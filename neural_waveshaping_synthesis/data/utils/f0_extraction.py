from functools import partial
from typing import Callable, Optional, Sequence, Union

import gin
import librosa
import numpy as np
import torch
import torchcrepe

from .upsampling import linear_interpolation
from ...utils import apply
from pathlib import Path


CREPE_WINDOW_LENGTH = 1024

def _get_f0_estimate_from_di(
    audio: np.ndarray,
    file: str,
    sample_rate: float,
    hop_length: int = 128,
    minimum_frequency: float = 50.0,
    maximum_frequency: float = 2000.0,
    full_model: bool = True,
    batch_size: int = 2048,
    device: Union[str, torch.device] = "cpu",
    interpolate_fn: Optional[Callable] = linear_interpolation,
    f0_from_di: bool = False
):


    """Add fundamental frequency (f0) estimate using CREPE."""
    global di_f0_estimates
    logging.debug(f"In _get_f0_estimate_from_di, keys of example: {ex.keys()}")
    filename = ex['audio_path'].split('/')[-1]
    passage = filename.split(" ")[-2]
    if not passage.isnumeric():
        raise Exception('Exception while parsing the passage number')
    di_audio_path = "/".join(ex['audio_path'].split("/")[:-1]) + f"/09A DI - {passage} .wav"
    if di_audio_path not in di_f0_estimates:
        logging.debug(f"keys of di_f0_estimates: {di_f0_estimates.keys()}")
        logging.debug(f"DI's corresponding f0 estimate not found for file {filename}, so calculating...")
        di_audio = _load_audio_as_array(di_audio_path, CREPE_SAMPLE_RATE)
        padding = 'center' if center else 'same'
        f0_hz, f0_confidence = spectral_ops.compute_f0(
            di_audio, frame_rate, viterbi=viterbi, padding=padding)
        di_f0_estimates[di_audio_path] = {
            'f0_hz': f0_hz,
            'f0_confidence': f0_confidence
        }
    else:
        f0_hz = di_f0_estimates[di_audio_path]['f0_hz']
        f0_confidence = di_f0_estimates[di_audio_path]['f0_confidence']

    return f0_hz, f0_confidence

@gin.configurable
def extract_f0_with_crepe(
    audio: np.ndarray,
    file: str,
    sample_rate: float,
    hop_length: int = 128,
    minimum_frequency: float = 50.0,
    maximum_frequency: float = 2000.0,
    full_model: bool = True,
    batch_size: int = 2048,
    device: Union[str, torch.device] = "cpu",
    interpolate_fn: Optional[Callable] = linear_interpolation,
    f0_from_di: bool = False
):
    # convert to torch tensor with channel dimension (necessary for CREPE)
    audio = torch.tensor(audio).unsqueeze(0)
    if f0_from_di:
        print("Getting f0 estimate from DI")
        file = Path(file)
        fn = file.name
        di_filename =  
        f0, confidence =  _get_f0_estimate_from_di(
            ex, frame_rate, center, viterbi
        )
    else:
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
