from functools import partial
from typing import Callable, Sequence, Union

import gin
import librosa
import numpy as np
import resampy
import scipy.io.wavfile as wavfile
from pathlib import Path

from .f0_extraction import extract_f0_with_crepe, extract_f0_with_pyin
from .loudness_extraction import extract_perceptual_loudness, extract_rms
from .mfcc_extraction import extract_mfcc
from ...utils import apply, apply_unpack, unzip

di_f0_estimates = {}


def read_audio_files(files: list):
    rates_and_audios = apply(wavfile.read, files)
    return unzip(rates_and_audios)


def convert_to_float32_audio(audio: np.ndarray):
    if audio.dtype == np.float32:
        return audio

    max_sample_value = np.iinfo(audio.dtype).max
    floating_point_audio = audio / max_sample_value
    return floating_point_audio.astype(np.float32)


def make_monophonic(audio: np.ndarray, strategy: str = "keep_left"):
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


def normalise_signal(audio: np.ndarray, factor: float):
    return audio / factor


def resample_audio(audio: np.ndarray, original_sr: float, target_sr: float):
    return resampy.resample(audio, original_sr, target_sr)


def segment_signal(
    signal: np.ndarray,
    sample_rate: float,
    segment_length_in_seconds: float,
    hop_length_in_seconds: float,
):
    segment_length_in_samples = int(sample_rate * segment_length_in_seconds)
    hop_length_in_samples = int(sample_rate * hop_length_in_seconds)
    segments = librosa.util.frame(
        signal, segment_length_in_samples, hop_length_in_samples
    )
    print(f"segment shape: {segments.shape}")
    return segments


def filter_segments(
    threshold: float,
    key_segments: np.ndarray,
    segments: Sequence[np.ndarray],
):
    mean_keys = key_segments.mean(axis=0)
    mask = mean_keys > threshold
    print(f"mean pitch conf: {mean_keys}")
    filtered_segments = apply(
        lambda x: x[:, mask] if len(x.shape) == 2 else x[:, :, mask], segments
    )
    return filtered_segments


def preprocess_single_audio_file(
    file: str,
    control_decimation_factor: float,
    target_sr: float = 16000.0,
    segment_length_in_seconds: float = 4.0,
    hop_length_in_seconds: float = 2.0,
    confidence_threshold: float = 0.85,
    f0_extractor: Callable = extract_f0_with_crepe,
    loudness_extractor: Callable = extract_perceptual_loudness,
    mfcc_extractor: Callable = extract_mfcc,
    normalisation_factor: Union[float, None] = None,
    f0_from_di: bool = False
):
    global di_f0_estimates
    print("\nLoading audio file: %s..." % file)
    original_sr, audio = wavfile.read(file)
    print(f"sr: {original_sr}, {audio.shape}")
    audio = convert_to_float32_audio(audio)
    audio = make_monophonic(audio)

    if normalisation_factor:
        audio = normalise_signal(audio, normalisation_factor)

    print("Resampling audio file: %s..." % file)
    print(f"audio.shape: {audio.shape}")

    audio = resample_audio(audio, original_sr, target_sr)

    if f0_from_di:
        print("Replacing f0 from DI...")

        file = Path(file)
        di_file = file.parent / f"09A DI - {file.name.split()[-1].split('.')[0]}.wav"
        if not di_file.exists():
            raise Exception(f"DI not found at {di_path}")

        if di_file.name in di_f0_estimates:
            f0, confidence = di_f0_estimates[di_filename]
        else:
            print("Loading DI file: %s..." % di_file)
            di_original_sr, di_audio = wavfile.read(di_file)
            di_audio = convert_to_float32_audio_dupli(di_audio)
            di_audio = make_monophonic_dupli(di_audio)

            if normalisation_factor:
                di_audio = normalise_signal_dupli(di_audio, normalisation_factor)

            print("Resampling audio file: %s..." % di_file)
            print(f"audio.shape: {di_audio.shape}")

            di_audio = resample_audio_dupli(di_audio, di_original_sr, target_sr)

            print("Extracting DI f0")
            f0, confidence = f0_extractor(audio=di_audio, normalisation_factor=normalisation_factor, target_sr=target_sr)
            di_f0_estimates[di_file.name] = [f0, confidence]
    else:
        print("Extracting f0 with extractor '%s': %s..." % (f0_extractor.__name__, file))
        f0, confidence = f0_extractor(audio=audio, file=file, normalisation_factor=normalisation_factor, target_sr=target_sr)

    print(f"f0.shape: {f0.shape}")

    print(f"%samples with conf > 0.95: {len([i for i in confidence if i >= 0.95])/len(confidence)*100}%")
    print(f"%samples with conf > 0.95: {len([i for i in confidence if i >= 0.90 and i < 0.95])/len(confidence)*100}%")
    print(f"%samples with conf > 0.95: {len([i for i in confidence if i >= 0.85 and i < 0.9])/len(confidence)*100}%")
    print(f"%samples with conf > 0.95: {len([i for i in confidence if i < 0.85])/len(confidence)*100}%")

    print(
        "Extracting loudness with extractor '%s': %s..."
        % (loudness_extractor.__name__, file)
    )
    loudness = loudness_extractor(audio)

    print(
        "Extracting MFCC with extractor '%s': %s..." % (mfcc_extractor.__name__, file)
    )
    mfcc = mfcc_extractor(audio)
    print(f"mfcc.shape: {mfcc.shape}")

    print("Segmenting audio file: %s..." % file)
    segmented_audio = segment_signal(
        audio, target_sr, segment_length_in_seconds, hop_length_in_seconds
    )
    print(f"segmented_audio.shape: {segmented_audio.shape}")

    print("Segmenting control signals: %s..." % file)
    print(f"control_decimation_factor: {control_decimation_factor}")
    segmented_f0 = segment_signal(
        f0,
        target_sr / (control_decimation_factor or 1),
        segment_length_in_seconds,
        hop_length_in_seconds,
    )
    print(f"segmented_f0.shape: {segmented_f0.shape}")

    segmented_confidence = segment_signal(
        confidence,
        target_sr / (control_decimation_factor or 1),
        segment_length_in_seconds,
        hop_length_in_seconds,
    )
    print(f"segmented_confidence.shape: {segmented_confidence.shape}")

    segmented_loudness = segment_signal(
        loudness,
        target_sr / (control_decimation_factor or 1),
        segment_length_in_seconds,
        hop_length_in_seconds,
    )
    print(f"segmented_loudness.shape: {segmented_loudness.shape}")

    segmented_mfcc = segment_signal(
        mfcc,
        target_sr / (control_decimation_factor or 1),
        segment_length_in_seconds,
        hop_length_in_seconds,
    )
    print(f"segmented_mfcc.shape: {segmented_mfcc.shape}")


    (
        filtered_audio,
        filtered_f0,
        filtered_confidence,
        filtered_loudness,
        filtered_mfcc,
    ) = filter_segments(
        confidence_threshold,
        segmented_confidence,
        (
            segmented_audio,
            segmented_f0,
            segmented_confidence,
            segmented_loudness,
            segmented_mfcc,
        ),
    )
    print(f"filtered_audio.shape: {filtered_audio.shape}")

    if filtered_audio.shape[-1] == 0:
        print("No segments exceeding confidence threshold...")
        audio_split, f0_split, confidence_split, loudness_split, mfcc_split = (
            [],
            [],
            [],
            [],
            [],
        )
    else:
        split = lambda x: [e.squeeze() for e in np.split(x, x.shape[-1], -1)]
        audio_split = split(filtered_audio)
        f0_split = split(filtered_f0)
        confidence_split = split(filtered_confidence)
        loudness_split = split(filtered_loudness)
        mfcc_split = split(filtered_mfcc)

    return audio_split, f0_split, confidence_split, loudness_split, mfcc_split


@gin.configurable
def preprocess_audio(
    files: list,
    control_decimation_factor: float,
    target_sr: float = 16000,
    segment_length_in_seconds: float = 4.0,
    hop_length_in_seconds: float = 2.0,
    confidence_threshold: float = 0.85,
    f0_extractor: Callable = extract_f0_with_crepe,
    loudness_extractor: Callable = extract_perceptual_loudness,
    normalise_audio: bool = False,
    f0_from_di: bool = True
):
    if normalise_audio:
        print("Finding normalisation factor...")
        normalisation_factor = 0
        for file in files:
            _, audio = wavfile.read(file)
            audio = convert_to_float32_audio(audio)
            audio = make_monophonic(audio)
            max_value = np.abs(audio).max()
            normalisation_factor = (
                max_value if max_value > normalisation_factor else normalisation_factor
            )

    # Check if dataset is timbre dataset

    processor = partial(
        preprocess_single_audio_file,
        control_decimation_factor=control_decimation_factor,
        target_sr=target_sr,
        segment_length_in_seconds=segment_length_in_seconds,
        hop_length_in_seconds=hop_length_in_seconds,
        confidence_threshold=confidence_threshold,
        f0_extractor=f0_extractor,
        loudness_extractor=loudness_extractor,
        normalisation_factor=None if not normalise_audio else normalisation_factor,
        f0_from_di=f0_from_di
    )
    for file in files:
        yield processor(file)
