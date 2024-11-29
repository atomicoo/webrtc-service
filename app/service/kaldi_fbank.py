from typing import Tuple
import numpy as np
import torch
import torchaudio

EPSILON = np.finfo(np.float32).eps
MILLISECONDS_TO_SECONDS = 0.001

def _next_power_of_2(x: int) -> int:
    r"""Returns the smallest power of 2 that is greater than x"""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def _get_strided(waveform: np.ndarray, window_size: int, window_shift: int) -> np.ndarray:
    assert waveform.ndim == 1
    num_samples = waveform.shape[0]
    strides = (waveform.strides[0] * window_shift, waveform.strides[0])

    if num_samples < window_size:
        return np.empty((0, 0), dtype=waveform.dtype)
    else:
        m = 1 + (num_samples - window_size) // window_shift

    sizes = (m, window_size)
    return np.lib.stride_tricks.as_strided(waveform, shape=sizes, strides=strides)

def _get_window(
    waveform: np.ndarray,
    padded_window_size: int,
    window_size: int,
    window_shift: int,
    remove_dc_offset: bool,
    preemphasis_coefficient: float,
) -> np.ndarray:
    # size (m, window_size)
    strided_input = _get_strided(waveform, window_size, window_shift)

    if remove_dc_offset:
        # Subtract each row/frame by its mean
        row_means = np.mean(strided_input, axis=1, keepdims=True)  # size (m, 1)
        strided_input = strided_input - row_means

    if preemphasis_coefficient != 0.0:
        # strided_input[i,j] -= preemphasis_coefficient * strided_input[i, max(0, j-1)] for all i,j
        offset_strided_input = np.pad(strided_input, ((0, 0), (1, 0)), mode='edge')  # size (m, window_size + 1)
        strided_input = strided_input - preemphasis_coefficient * offset_strided_input[:, :-1]

    # Apply window_function to each row/frame
    window_function = np.hanning(window_size) ** 0.85  # size (window_size,)
    strided_input = strided_input * window_function  # size (m, window_size)

    # Pad columns with zero until we reach size (m, padded_window_size)
    if padded_window_size != window_size:
        padding_right = padded_window_size - window_size
        strided_input = np.pad(strided_input, ((0, 0), (0, padding_right)), mode='constant', constant_values=0)

    return strided_input

def get_mel_banks(
    num_bins: int,
    window_length_padded: int,
    sample_freq: float,
    low_freq: float,
    high_freq: float,
) -> Tuple[np.ndarray, np.ndarray]:
    assert num_bins > 3, "Must have at least 3 mel bins"
    assert window_length_padded % 2 == 0
    num_fft_bins = window_length_padded // 2

    # fft-bin width [think of it as Nyquist-freq / half-window-length]
    fft_bin_width = sample_freq / window_length_padded
    mel_low_freq = mel_scale_scalar(low_freq)
    mel_high_freq = mel_scale_scalar(high_freq)

    # divide by num_bins+1 in next line because of end-effects where the bins
    # spread out to the sides.
    mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)

    bin = np.arange(num_bins).reshape(num_bins, 1)
    left_mel = mel_low_freq + bin * mel_freq_delta  # size(num_bins, 1)
    center_mel = mel_low_freq + (bin + 1.0) * mel_freq_delta  # size(num_bins, 1)
    right_mel = mel_low_freq + (bin + 2.0) * mel_freq_delta  # size(num_bins, 1)

    center_freqs = inverse_mel_scale(center_mel)  # size (num_bins)
    # size(1, num_fft_bins)
    mel = mel_scale(fft_bin_width * np.arange(num_fft_bins)).reshape(1, num_fft_bins)

    # size (num_bins, num_fft_bins)
    up_slope = (mel - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel) / (right_mel - center_mel)

    # left_mel < center_mel < right_mel so we can min the two slopes and clamp negative values
    bins = np.maximum(0, np.minimum(up_slope, down_slope))

    return bins, center_freqs

def fbank(
    waveform: np.ndarray,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
    high_freq: float = 8000.0,
    low_freq: float = 20.0,
    min_duration: float = 0.0,
    num_mel_bins: int = 23,
    preemphasis_coefficient: float = 0.97,
    remove_dc_offset: bool = True,
    round_to_power_of_two: bool = True,
    sample_frequency: float = 16000.0,
):

    window_shift = int(sample_frequency * frame_shift * MILLISECONDS_TO_SECONDS)
    window_size = int(sample_frequency * frame_length * MILLISECONDS_TO_SECONDS)
    padded_window_size = _next_power_of_2(window_size) if round_to_power_of_two else window_size

    if len(waveform) < min_duration * sample_frequency:
        return np.empty(0, dtype=np.float32)

    # strided_input, size (m, padded_window_size) and signal_log_energy, size (m)
    strided_input = _get_window(
        waveform,
        padded_window_size,
        window_size,
        window_shift,
        remove_dc_offset,
        preemphasis_coefficient,
    )

    # size (m, padded_window_size // 2 + 1)
    spectrum = np.abs(np.fft.rfft(strided_input))
    spectrum = np.power(spectrum, 2.0)

    # size (num_mel_bins, padded_window_size // 2)
    mel_energies, _ = get_mel_banks(num_mel_bins, padded_window_size, sample_frequency, low_freq, high_freq)
    # pad right column with zeros and add dimension, size (num_mel_bins, padded_window_size // 2 + 1)
    mel_energies = np.pad(mel_energies, ((0, 0), (0, 1)), mode='constant', constant_values=0)
    # sum with mel fiterbanks over the power spectrum, size (m, num_mel_bins)
    mel_energies = np.dot(spectrum, mel_energies.T)
    # avoid log of zero (which should be prevented anyway by dithering)
    mel_energies = np.log(np.maximum(mel_energies, EPSILON))

    return mel_energies

def inverse_mel_scale_scalar(mel_freq: float) -> float:
    return 700.0 * (np.exp(mel_freq / 1127.0) - 1.0)

def inverse_mel_scale(mel_freq: np.ndarray) -> np.ndarray:
    return 700.0 * (np.exp(mel_freq / 1127.0) - 1.0)

def mel_scale_scalar(freq: float) -> float:
    return 1127.0 * np.log(1.0 + freq / 700.0)

def mel_scale(freq: np.ndarray) -> np.ndarray:
    return 1127.0 * np.log(1.0 + freq / 700.0)


if __name__ == '__main__':
    signal = np.random.randn(16000)  # Replace this with your actual signal
    sample_rate = 16000  # Replace this with your actual sample rate
    num_mel_bins = 80
    frame_length = 25  # in milliseconds
    frame_shift = 10  # in milliseconds

    # Compute fbank features using torchaudio for comparison
    fbank_features = fbank(signal,
                        num_mel_bins=num_mel_bins,
                        frame_length=frame_length,
                        frame_shift=frame_shift,
                        sample_frequency=sample_rate)
    print(torch.from_numpy(fbank_features).to(dtype=torch.float32))

    torch_signal = torch.tensor(signal, dtype=torch.float32)
    fbank_features_torchaudio = torchaudio.compliance.kaldi.fbank(torch_signal.unsqueeze(0),
                            num_mel_bins=num_mel_bins,
                            frame_length=frame_length,
                            frame_shift=frame_shift,
                            dither=0,
                            energy_floor=0.0,
                            sample_frequency=sample_rate)
    print(fbank_features_torchaudio)