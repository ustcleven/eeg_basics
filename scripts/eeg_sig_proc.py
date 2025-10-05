import mne
import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy.signal import welch
from scipy.signal import stft
from eeg_band_tracker import STFTConfig, RunningBaseline, stft_power_1shot, band_indices, aggregate_bands

# ---------------------------
# 1) Helper: compute alpha band power (V^2) via Welch
# ---------------------------
def compute_alpha_power(signal, fs, band=(8, 12), nperseg=None, noverlap=None, window='hann'):
    """
    Compute band power using Welch PSD and integrate over the band (V^2).
    signal: 1D numpy array in Volts
    fs: sampling rate (Hz)
    """
    if nperseg is None:
        nperseg = int(2 * fs)  # 2-second windows (good for alpha)
    if noverlap is None:
        noverlap = nperseg // 2  # 50% overlap
    f, Pxx = welch(signal, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, scaling='density')
    fmin, fmax = band
    mask = (f >= fmin) & (f <= fmax)
    power = np.trapz(Pxx[mask], f[mask])  # integrate PSD over band -> V^2
    return power

def plot_front_lobe_asymmetry(input_path):
    # ---------------------------
    # 2) Load EEG (replace path)
    # ---------------------------
    raw = mne.io.read_raw_edf(input_path, preload=True)  # <-- set your path
    fs = raw.info['sfreq']

    # (Optional but recommended) light filtering for alpha analyses:
    # raw = raw.copy().filter(l_freq=1., h_freq=40., picks='eeg')

    # Choose frontal channels of interest (some files may not have all)
    candidate_chs = ['Fp1.','Fp2.','F3..','F4..','F7..','F8..','Fc3.','Fc4.']

    # Build signals dict from channels that exist
    signals = {}
    avail = set(raw.ch_names)
    for ch in candidate_chs:
        if ch in avail:
            sig, _ = raw.get_data(picks=[ch], return_times=True)
            signals[ch] = sig.flatten()
        else:
            print(f"[warn] Channel {ch} not found in this recording; skipping.")

    # Require at least one valid pair
    pairs = [('Fp1.','Fp2.'), ('F3..','F4..'), ('F7..','F8..'), ('Fc3.','Fc4.')]
    pairs = [(L,R) for (L,R) in pairs if (L in signals and R in signals)]
    if not pairs:
        raise RuntimeError("None of the target frontal pairs are present in this file.")

    # ---------------------------
    # 3) Compute alpha power & FAA per pair
    # ---------------------------
    alpha_powers = {}
    for ch, sig in signals.items():
        alpha_powers[ch] = compute_alpha_power(sig, fs, band=(8,12))

    # FAA = ln(alpha_right) - ln(alpha_left)
    faa_values = { (L,R): (np.log(alpha_powers[R] + 1e-20) - np.log(alpha_powers[L] + 1e-20))
                for (L,R) in pairs }

    # ---------------------------
    # 4A) Plot 1 — FAA bar plot (per pair)
    # ---------------------------
    labels = [f"{L}-{R}" for (L,R) in pairs]
    faa_bar = [faa_values[(L,R)] for (L,R) in pairs]

    plt.figure(figsize=(8,4), num=1)
    plt.bar(labels, faa_bar)
    plt.axhline(0, color='k', lw=1)
    plt.ylabel("FAA = ln(alpha_R) - ln(alpha_L)")
    plt.title("Frontal Alpha Asymmetry (FAA)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # ---------------------------
    # 4B) Plot 2 — Simple head map with frontal electrodes
    # Color = log alpha power; dashed lines connect asymmetry pairs
    # ---------------------------

    # Approximate 2D positions (10–20-ish) for illustration; adjust if you like
    electrode_pos = {
        'Fp1.': (-0.45,  0.85), 'Fp2.': ( 0.45,  0.85),
        'F7..' : (-0.75,  0.45), 'F8..' : ( 0.75,  0.45),
        'F3..' : (-0.40,  0.55), 'F4..' : ( 0.40,  0.55),
        'Fc3.': (-0.30,  0.25), 'Fc4.': ( 0.30,  0.25),
    }

    # Keep only positions for channels we actually have
    plot_chs = [ch for ch in electrode_pos if ch in signals]
    vals = np.array([np.log(alpha_powers[ch] + 1e-20) for ch in plot_chs])
    vmin, vmax = vals.min(), vals.max()

    fig, ax = plt.subplots(figsize=(6,6), num=2)

    # head outline
    theta = np.linspace(0, 2*np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), 'k', lw=2)
    # nose
    ax.plot([0, -0.08, 0.08, 0], [1.0, 1.1, 1.1, 1.0], 'k', lw=2)
    # ears
    ax.plot([ -1.02, -1.05, -1.02], [0.2, 0.0, -0.2], 'k', lw=2)
    ax.plot([  1.02,  1.05,  1.02], [0.2, 0.0, -0.2], 'k', lw=2)

    # scatter electrodes (color = log alpha power)
    sc = None
    for ch in plot_chs:
        x, y = electrode_pos[ch]
        val = np.log(alpha_powers[ch] + 1e-20)
        sc = ax.scatter(x, y, s=360, c=[[val]], vmin=vmin, vmax=vmax,
                        cmap='viridis', edgecolor='k')
        ax.text(x, y, ch, ha='center', va='center', color='w',
                fontsize=10, fontweight='bold')

    # dashed lines for pairs that exist
    for (L,R) in pairs:
        x1, y1 = electrode_pos.get(L, (None,None))
        x2, y2 = electrode_pos.get(R, (None,None))
        if x1 is not None and x2 is not None:
            ax.plot([x1, x2], [y1, y2], 'k--', alpha=0.6)

    ax.set_aspect('equal', 'box')
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.1, 1.2)
    ax.axis('off')
    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.03)
        cbar.set_label("log alpha power (V²)")
    ax.set_title("Frontal electrodes: log alpha power + asymmetry pairs")
    plt.tight_layout()
    plt.show()

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
        plot_aggreated_bands(input_path)
    elif test_type =="plot_front_lobe":
        plot_front_lobe_asymmetry(input_path)


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

