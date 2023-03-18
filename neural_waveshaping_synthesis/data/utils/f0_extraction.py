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
di_f0_estimates = {}

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
        di_filename = f"09A DI - {file.name.split()[-1].split('.').strip()}.wav"
        di_path = file.parent / di_filename
        if not di_path.exists():
            raise Exception(f"DI not found at {di_path}")

        print("\nLoading DI file: %s..." % di_path)
        original_sr, di_audio = wavfile.read(di_path)
        di_audio = convert_to_float32_audio(di_audio)
        di_audio = make_monophonic(di_audio)

        if normalisation_factor:
            audio = normalise_signal(di_audio, normalisation_factor)

        print("Resampling audio file: %s..." % file)
        print(f"audio.shape: {audio.shape}")

        audio = resample_audio(audio, original_sr, target_sr)

        print("Extracting f0 with extractor '%s': %s..." % (f0_extractor.__name__, file))
        f0, confidence = f0_extractor(audio, file)

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
