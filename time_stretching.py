from scipy.fft import rfft, irfft
from scipy.io import wavfile

import numpy as np
import argparse


def hann(N):
    """
    Hanning window

    Parameters
    ----------
    N (int): window size
    """

    n = np.arange(0, N)
    return 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))


def phase_vocoder(
    input_data,
    sample_rate,
    r,
    window_size=1024,
    ):
    """
    Phase vocoder algorithm

    Parameters
    ----------
    input_data (numpy.ndarray): input signal
    sample_rate (int): signal frequency
    r (float): time stretch ratio (0 < r < 1 - squeezing, 1 <= r - stretching)
    window_size (int, optinal): window size
    """

    # Analysis

    hop_a = window_size // 4
    hop_s = int(np.floor(hop_a * r))
    w = hann(window_size)
    n_window = input_data.size // hop_a + 1 - 3

    windows = np.zeros((n_window, window_size // 2 + 1), dtype=complex)
    n_zeros = (n_window - 1) * hop_a + window_size - input_data.size
    padding_data = np.append(input_data, np.zeros(n_zeros))

    for i in range(n_window):
        windows[i] = rfft(padding_data[i * hop_a:i * hop_a
                          + window_size] * w)

    # Processing and Synthesis

    w_bin = sample_rate / (window_size // 2 + 1) * np.arange(0,
            window_size // 2 + 1)
    (cur_phase, last_phase) = (np.zeros(window_size),
                               np.angle(windows[0, :]))
    q = np.zeros((n_window, window_size))
    q[0, :] = irfft(np.exp(1j * last_phase) * np.abs(windows[0, :])) * w
    output_data = np.zeros((n_window - 1) * hop_s + window_size)

    for i in range(1, n_window):
        w_delta = (np.angle(windows[i]) - np.angle(windows[i - 1])) \
            / hop_a * sample_rate - w_bin
        w_delta_wrapped = (w_delta + np.pi) % (2 * np.pi) - np.pi
        w_true = w_bin + w_delta_wrapped

        cur_phase = last_phase + hop_s / hop_a * w_true
        q[i, :] = irfft(np.exp(1j * cur_phase) * np.abs(windows[i])) * w

    for i in range(n_window):
        output_data[i * hop_s:i * hop_s + window_size] += q[i, :]

    output_data = output_data / np.max(np.abs(output_data)) \
        * np.max(np.abs(input_data))
    output_data = output_data.astype(type(input_data[0]))

    return output_data


def time_stretching(input_wav, output_wav, r):
    """
    Function for stretching or squeezing wav audio file without changing pitch.

    Parameters
    ----------
    input_wav (string): path to the input wav audio file
    output_wav (string): path to save the converted file
    r (float): time stretch ratio (0 < r < 1 - squeezing, 1 <= r - stretching)
    """

    # Reading input:

    (sample_rate, input_data) = wavfile.read(input_wav)

    output_data = np.array(0)
    if len(input_data.shape) == 1:
        output_data = phase_vocoder(input_data=input_data,
                                    sample_rate=sample_rate, r=r)
    else:
        for i in range(input_data.shape[1]):
            result = phase_vocoder(input_data=input_data[:, i],
                                   sample_rate=sample_rate, r=r)
            if output_data.size == 1:
                output_data = result[:, None]
            else:
                output_data = np.concatenate((output_data, result[:,
                        None]), axis=1)

    # Saving result

    wavfile.write(output_wav, sample_rate, output_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default='')
    parser.add_argument('--output-path', type=str, default='')
    parser.add_argument('--r', type=float, default=1.0)
    args = parser.parse_args()

    time_stretching(args.input_path, args.output_path, args.r)