import mne
import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy.signal import welch

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
    plt.plot(times, data[0]*1e6, label='F3')   # convert V → µV
    plt.plot(times, data[1]*1e6, label='F4')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
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
    plt.show()

    # plt.figure(101)
    # ax1 = plt.subplot(2,1,1)
    # ax1 = plt.gca()
    # ax1.plot(times, data[0]*1e6, label='F3')
    # ax1.plot(times, data[1]*1e6, label='F4')
    # ax1.legend(loc='upper center', bbox_to_anchor=(0.9, 0.9),ncol=1, fancybox=True, shadow=True)
    # ax1.set_xlabel("Time (s)")
    # ax1.set_ylabel("Amplitude (µV)")
    # ax1.set_title("EEG F3/F4")   
    # ax1.grid()       
    # ax2 = plt.subplot(2,1,2, sharex=ax1)
    # ax2.plot(times, 10*np.log10(data[0]*data[0]*1e12), label='F3')
    # # ax2.plot(times, data[1]*1e6, label='F4')
    # ax2 = plt.gca()
    # ax2.legend(loc='upper center', bbox_to_anchor=(0.9, 0.9),ncol=1, fancybox=True, shadow=True)
    # ax1.set_xlabel("Time (s)")
    # ax1.set_ylabel("Amplitude dB (V)")
    # ax2.grid()     
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

