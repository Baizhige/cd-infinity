import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import os


def calculate_avg_spectrum(data, sampling_rate):
    """
    Calculate the average frequency spectrum of the data.
    """
    # FFT transform
    fft_data = fft(data, axis=2)
    # Calculate magnitude
    mag = np.abs(fft_data)
    # Average over samples and channels
    avg_mag = np.mean(mag, axis=(0, 1))
    # Frequency axis
    freqs = np.fft.fftfreq(data.shape[2], 1 / sampling_rate)
    # Only keep the positive frequencies
    pos_freqs = freqs[:len(freqs) // 2]
    pos_avg_mag = avg_mag[:len(freqs) // 2]
    return pos_freqs, pos_avg_mag


def draw_fre_mag(stages_list, sampling_rate):
    n_stages = len(stages_list)
    fig, axes = plt.subplots(n_stages, 4, figsize=(20, 5 * n_stages), dpi=300)

    for i, stage in enumerate(stages_list):
        # Load data
        source_path = os.path.join("record", f"source_{stage}_transformed_data.npy")
        target_path = os.path.join("record", f"target_{stage}_transformed_data.npy")
        source_data = np.load(source_path)
        target_data = np.load(target_path)

        # Calculate average spectrum
        source_freqs, source_avg_mag = calculate_avg_spectrum(source_data, sampling_rate)
        target_freqs, target_avg_mag = calculate_avg_spectrum(target_data, sampling_rate)

        # Frequency bands
        bands = [(0, 4), (4, 8), (8, 13), (13, 30)]

        for j, (fmin, fmax) in enumerate(bands):
            ax = axes[i, j] if n_stages > 1 else axes[j]
            # Find indexes for the frequency band
            idx = np.logical_and(source_freqs >= fmin, source_freqs <= fmax)
            # Plot source
            ax.plot(source_freqs[idx], source_avg_mag[idx], color='red', label=f"Source {stage}")
            # Plot target
            ax.plot(target_freqs[idx], target_avg_mag[idx], color='blue', label=f"Target {stage}")
            # Adjust y-axis range
            ax.autoscale(enable=True, axis='y', tight=True)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Magnitude')
            ax.legend()
            ax.grid(True, which='both', axis='both')
            ax.minorticks_on()

    plt.tight_layout()
    plt.savefig(os.path.join('figures','average_spectrum.pdf'), format='pdf')


# Example usage
stages_list = ["prior", "adjust", "FIR"]
sampling_rate = 128
draw_fre_mag(stages_list, sampling_rate)