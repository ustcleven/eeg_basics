import mne
import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy.signal import welch
from scipy.signal import stft
from eeg_band_tracker import STFTConfig, RunningBaseline, stft_power_1shot, band_indices, aggregate_bands

def plot_aggreated_bands(input_path):
    raw = mne.io.read_raw_edf(input_path, preload=True)
    raw.filter(l_freq=1.0, h_freq=70.0)
    raw.notch_filter(freqs=[60])

    # Get data for F3 and F4
    data, times = raw.get_data(picks=['F3..', 'F4..'], return_times=True)
    fs = raw.info['sfreq']  # sampling frequency
    f, pxx_f3 = welch(data[0], fs=fs, nperseg=1024)
    plt.figure(num=2)
    plt.semilogy(f, pxx_f3, label='F3')
    plt.xlim(0, 40)  # EEG band
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (V^2/Hz)")
    plt.grid()
    plt.legend()
    # print(data.shape())
    x = data[0]
    # Inputs: x = 1D numpy EEG for one channel; fs = sampling rate
    fs = 160
    cfg = STFTConfig(fs=fs, nperseg=320, noverlap=240, window="hann")

    # Warm-up to get frequency grid and init baseline
    fgrid, _ = stft_power_1shot(np.zeros(cfg.nperseg), cfg)
    rb = RunningBaseline(n_freq=fgrid.size, alpha=0.95)

    BANDS = {"delta": (1,2.5), "theta": (4,8), "alpha": (12,13), "beta": (13,30)}
    bidx = band_indices(fgrid, BANDS)

    # Stream your data in chunks of length nperseg with hop = nperseg - noverlap
    hop = cfg.nperseg - cfg.noverlap
    n_steps = (len(x) - cfg.nperseg) // hop + 1
    band_names = list(BANDS.keys())
    heat = np.full((len(band_names), n_steps), np.nan)

    for i in range(n_steps):
        start = i * hop
        chunk = x[start:start+cfg.nperseg]
        f, P_lin = stft_power_1shot(chunk, cfg)

        # update + compute per-frequency contrast (in dB)
        rb.update(P_lin)
        contrast_db = rb.contrast_db(P_lin)

        # collapse frequency → band
        vals = aggregate_bands(contrast_db, bidx)
        heat[:, i] = [vals[b] for b in band_names]

    # Final one-shot plot
    # convert hop index → time (seconds)
    hop_len = cfg.nperseg - cfg.noverlap
    time_axis = np.arange(n_steps) * hop_len / fs  # seconds

    plt.figure(figsize=(10, 4), num=5)
    plt.imshow(
        heat,
        aspect='auto',
        origin='lower',
        extent=[time_axis[0], time_axis[-1], 0, len(band_names)],
    )
    plt.yticks(np.arange(len(band_names)) + 0.5, band_names)
    plt.xlabel("Time (s)")
    plt.title("Baseline-relative band contrast (dB)")
    plt.colorbar(label="dB")
    plt.tight_layout()
    plt.show()


def plot_eeg_basic(input_path):
    # Example: load sample EEG file (replace with your own path)
    raw = mne.io.read_raw_edf(input_path, preload=True)

    print(raw.ch_names)
    # # # montage = mne.channels.make_standard_montage('standard_1020')
    # # # mapping = {
    # # #     'Fp1': 'Fp1', 'Fp2': 'Fp2',  'Fz': 'Fz', 'Cz': 'Cz',
    # # #     'CPz': 'CPz', 'Pz': 'Pz', 'POz': 'POz', 'Oz': 'Oz'
    # # # }
    # # # raw.rename_channels(mapping)
    # # # raw.set_channel_types({'VEOG': 'eog'})
    # # # raw.set_montage(montage)

    # # # Pick one channel to plot
    # # raw.plot(n_channels=2, duration=10, scalings='auto')
    # # raw.plot_psd(fmax=50, show=True)

    # Get data for F3 and F4
    data, times = raw.get_data(picks=['F3..', 'F4..'], return_times=True)

    plt.figure(num=1)
    plt.plot(times, data[0]*1e6, label='F3')   # convert V to microV
    plt.plot(times, data[1]*1e6, label='F4')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (microV)")
    # plt.yticks([-100, -50, 0, 50, 100])  # custom y-ticks
    plt.legend()
    plt.grid()

    fs = raw.info['sfreq']  # sampling frequency
    f, pxx_f3 = welch(data[0], fs=fs, nperseg=1024)
    f, pxx_f4 = welch(data[1], fs=fs, nperseg=1024)

    plt.figure(num=2)
    plt.semilogy(f, pxx_f3, label='F3')
    plt.semilogy(f, pxx_f4, label='F4')
    plt.xlim(0, 40)  # EEG band
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (V^2/Hz)")
    plt.grid()
    plt.legend()

    pxx_f3_db = 10 * np.log10(pxx_f3)
    pxx_f4_db = 10 * np.log10(pxx_f4)

    # Plot
    plt.figure(figsize=(8,5), num=3)
    plt.plot(f, pxx_f3_db, label='F3')
    plt.plot(f, pxx_f4_db, label='F4')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (dB/Hz)")
    plt.title("Welch PSD for F3 and F4")
    plt.legend()
    plt.grid(True)

    # Suppose data_f3 and data_f4 are your two EEG channels (1D NumPy arrays)
    # Example: data_f3, data_f4 = raw.get_data(picks=['F3','F4'])

    # Compute STFT for both channels
    data_f3 = data[0]
    data_f4 = data[1]
    f_f3, t_f3, Zxx_f3 = stft(data_f3, fs=fs, nperseg=320, noverlap=240)
    f_f4, t_f4, Zxx_f4 = stft(data_f4, fs=fs, nperseg=320, noverlap=240)

    # Create figure with 2 rows (one for each channel)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), num=4)  # <-- figure number 3

    # ---- Plot F3 ----
    im1 = axes[0].pcolormesh(t_f3, f_f3, 10*np.log10(np.abs(Zxx_f3)**2), shading='gouraud')
    axes[0].set_title('STFT Spectrum - F3')
    axes[0].set_ylabel('Frequency [Hz]')
    axes[0].set_ylim(0, 40)
    fig.colorbar(im1, ax=axes[0], label='Power (dB)')

    # ---- Plot F4 ----
    im2 = axes[1].pcolormesh(t_f4, f_f4, 10*np.log10(np.abs(Zxx_f4)**2), shading='gouraud')
    axes[1].set_title('STFT Spectrum - F4')
    axes[1].set_ylabel('Frequency [Hz]')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylim(0, 40)
    fig.colorbar(im2, ax=axes[1], label='Power (dB)')

    # ---- Final adjustments ----
    plt.tight_layout()

    N_f3 = len(data_f3)
    N_f4 = len(data_f4)
    freqs_f3 = np.fft.rfftfreq(N_f3, d=1/fs)
    freqs_f4 = np.fft.rfftfreq(N_f4, d=1/fs)

    fft_f3 = np.fft.rfft(data_f3)
    fft_f4 = np.fft.rfft(data_f4)

    # Magnitude in dB
    mag_f3_db = 20 * np.log10(np.abs(fft_f3) + 1e-12)
    mag_f4_db = 20 * np.log10(np.abs(fft_f4) + 1e-12)

    # Create figure with two stacked subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), num=5)  # Figure number 5

    # ---- F3 ----
    axes[0].plot(freqs_f3, mag_f3_db, color='C0')
    axes[0].set_title('FFT Magnitude - F3')
    axes[0].set_xlim(0, 40)  # EEG band
    axes[0].set_ylim(-100, 0)
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].grid(True)

    # ---- F4 ----
    axes[1].plot(freqs_f4, mag_f4_db, color='C1')
    axes[1].set_title('FFT Magnitude - F4')
    axes[1].set_xlim(0, 40)
    axes[1].set_ylim(-100, 0)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude (dB)')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_notch_filter(input_path):
    # Example: load sample EEG file (replace with your own path)
    raw = mne.io.read_raw_edf(input_path, preload=True)

    # Get data for F3 and F4
    data, times = raw.get_data(picks=['F3..', 'F4..'], return_times=True)

    fs = raw.info['sfreq']  # sampling frequency
    print("sampling rate=", fs)
    f, pxx_f3 = welch(data[0], fs=fs, nperseg=1024)
    f, pxx_f4 = welch(data[1], fs=fs, nperseg=1024)

    pxx_f3_db = 10 * np.log10(pxx_f3)
    pxx_f4_db = 10 * np.log10(pxx_f4)

    # Plot
    plt.figure(figsize=(8,5), num=3)
    plt.plot(f, pxx_f3_db, label='F3')
    plt.plot(f, pxx_f4_db, label='F4')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (dB/Hz)")
    plt.title("Welch PSD for F3 and F4")
    plt.legend()
    plt.grid(True)

    raw.filter(l_freq=1.0, h_freq=70.0)
    raw.notch_filter(freqs=[60])
    data, times = raw.get_data(picks=['F3..', 'F4..'], return_times=True)

    fs = raw.info['sfreq']  # sampling frequency
    f, pxx_f3 = welch(data[0], fs=fs, nperseg=1024)
    f, pxx_f4 = welch(data[1], fs=fs, nperseg=1024)

    pxx_f3_db = 10 * np.log10(pxx_f3)
    pxx_f4_db = 10 * np.log10(pxx_f4)

    # Plot
    plt.figure(figsize=(8,5), num=4)
    plt.plot(f, pxx_f3_db, label='F3')
    plt.plot(f, pxx_f4_db, label='F4')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (dB/Hz)")
    plt.title("Welch PSD for F3 and F4 with filtering")
    plt.legend()
    plt.grid(True)


    plt.show()   


def main(test_type, debug_type, input_path):
    

    if test_type == "plot":
        plot_eeg_basic(input_path)
    elif test_type == "notch_filter":
        plot_notch_filter(input_path)
    elif test_type =="plot_bands":
        print("ttt")
        plot_aggreated_bands(input_path)
     


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t',
        '--test_type',
        type=str,
        default='test_fd_aic',
        help="""\
                test_fd_aic: test fd aic
                plot: plot debug resuls
                gen_header: generate c healders
                plot_c: plot c results
            """)

    parser.add_argument(
        '-d',
        '--debug_type',
        type=str,
        default="flt",
        help="""\
            algo_type: algos for generating c header
            """)

    parser.add_argument(
        '-f',
        '--input_path',
        type=str,
        default="../fixtures/eegmmidb/S001/S001R01.edf",
        help="""\
            input_path: eeg file path
            """)


    args, unparsed = parser.parse_known_args()

    test_type = args.test_type
    debug_type = args.debug_type
    input_path = args.input_path

    main(test_type, debug_type, input_path)

