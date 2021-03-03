import math

import librosa
import numpy as np
from scipy.fftpack import dct

# If you want to see the spectrogram picture
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
def plot_spectrogram(spec, note,file_name):
    """Draw the spectrogram picture
        :param spec: a feature_dim by num_frames array(real)
        :param note: title of the picture
        :param file_name: name of the file
    """ 
    fig = plt.figure(figsize=(20, 5))
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.xlabel('Time(s)')
    plt.ylabel(note)
    plt.tight_layout()
    plt.savefig(file_name)


#preemphasis config 
alpha = 0.97

# Enframe config
frame_len = 400      # 25ms, fs=16kHz
frame_shift = 160    # 10ms, fs=15kHz
fft_len = 512

# Mel filter config
num_filter = 23
num_mfcc = 12

# Read wav file
wav, fs = librosa.load('./test.wav', sr=None)

# Enframe with Hamming window function
def preemphasis(signal, coeff=alpha):
    """perform preemphasis on the input signal.

        :param signal: The signal to filter.
        :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.97.
        :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def enframe(signal, frame_len=frame_len, frame_shift=frame_shift, win=np.hamming(frame_len)):
    """Enframe with Hamming widow function.

        :param signal: The signal be enframed
        :param win: window function, default Hamming
        :returns: the enframed signal, num_frames by frame_len array
    """
    
    num_samples = signal.size
    num_frames = np.floor((num_samples - frame_len) / frame_shift)+1
    frames = np.zeros((int(num_frames),frame_len))
    for i in range(int(num_frames)):
        frames[i,:] = signal[i*frame_shift:i*frame_shift + frame_len] 
        frames[i,:] = frames[i,:] * win

    return frames

def get_spectrum(frames, fft_len=fft_len):
    """Get spectrum using fft
        :param frames: the enframed signal, num_frames by frame_len array
        :param fft_len: FFT length, default 512
        :returns: spectrum, a num_frames by fft_len/2+1 array (real)
    """
    cFFT = np.fft.fft(frames, n=fft_len)
    valid_len = int(fft_len / 2 ) + 1
    spectrum = np.abs(cFFT[:,0:valid_len])
    return spectrum

def fbank(spectrum, num_filter = num_filter):
    """Get mel filter bank feature from spectrum
        :param spectrum: a num_frames by fft_len/2+1 array(real)
        :param num_filter: mel filters number, default 23
        :returns: fbank feature, a num_frames by num_filter array 
        DON'T FORGET LOG OPRETION AFTER MEL FILTER!
    """

    # feats=np.zeros((spectrum.shape[0], num_filter))
    f = mel_filter(num_filter, 16000, fft_len)
    feats = np.dot(spectrum, f.T)
    feats = np.log10(feats)
    return feats

def mfcc(fbank, num_mfcc = num_mfcc):
    """Get mfcc feature from fbank feature
        :param fbank: a num_frames by  num_filter array(real)
        :param num_mfcc: mfcc number, default 12
        :returns: mfcc feature, a num_frames by num_mfcc array 
    """

    # feats = np.zeros((fbank.shape[0],num_mfcc))
    feats = dct(fbank)[:, 1:(num_mfcc + 1)]
    return feats

def write_file(feats, file_name):
    """Write the feature to file
        :param feats: a num_frames by feature_dim array(real)
        :param file_name: name of the file
    """
    f=open(file_name,'w')
    (row,col) = feats.shape
    for i in range(row):
        f.write('[')
        for j in range(col):
            f.write(str(feats[i,j])+' ')
        f.write(']\n')
    f.close()

def mel_filter(num_filter, fs, n):
    """
    :param num_filter: 梅尔滤波器个数
    :param fs: 采样频率
    :param n:  dft 点数
    """
    fh = fs / 2
    mel_fh = 2595 * np.log10(1 + fh / 700)
    x = np.linspace(0, mel_fh, num_filter + 2)  # mel 刻度
    for i in range(num_filter + 2):  # 索引值
        x[i] = math.floor(700 * (10 ** (x[i] / 2595) - 1) / fs * n)
    num_points = n // 2 + 1
    y = np.zeros((num_filter, num_points))
    for k in range(1, num_filter + 1):
        n0 = int(x[k - 1])
        n1 = int(x[k])
        n2 = int(x[k + 1])
        for i in range(n0, n1):
            y[k-1, i] = (i - n0) / (n1 - n0)
        for i in range(n1, n2):
            y[k-1, i] = (n2 - i) / (n2 - n1)
     
    # freqs = [int(fs / n * i) for i in range(num_points)]
    # for k in range(num_filter):
    #     plt.plot(freqs, y[k])
    # plt.show()
    return y

def plot_wav(wav, fs, i, j, k, label, color):
    t = len(wav) / fs * 1000
    x = np.arange(0, t, 1/fs * 1000)
    plt.subplot(i, j, k)
    plt.plot(x, wav, color)
    plt.xlabel(label)

def plot_spectrum(data, i, j, k, label, color):
    x = np.arange(0, len(data), 1)
    plt.subplot(i, j, k)
    plt.plot(x, data, color)
    plt.xlabel(label)

def fft(x):
    return np.abs(np.fft.fft(x, n=fft_len))

def test_every_step():
    wav, fs = librosa.load('./test.wav', sr=None)
    frame = wav[:400]
    frame_preemphasis = preemphasis(frame)
    win = np.hamming(len(frame))
    frame_window = frame_preemphasis * win
    frame_spectrum = fft(frame)[:257]
    frame_preemphasis_spectrum = fft(frame_preemphasis)[:257]
    frame_window_spectrum = fft(frame_window)[:257]
    fbank_feats = fbank(frame_window_spectrum)
    mfcc_feats = dct(fbank_feats)

    # 原始信号
    plot_wav(frame, fs, 5, 2, 1, "original", "blue")
    # 原始信号频谱
    plot_spectrum(frame_spectrum, 5, 2, 2, "original", "grey")
    # 预加重处理后的信号
    plot_wav(frame_preemphasis, fs, 5, 2, 3, "preemphasis", "blue")
    # 预加重处理后的信号频谱
    plot_spectrum(frame_preemphasis_spectrum, 5, 2, 4, "preemphasis", "grey")
    # 加窗信号
    plot_wav(frame_window, fs, 5, 2, 5, "window", "blue")
    # 加窗信号频谱
    plot_spectrum(frame_window_spectrum, 5, 2, 6, "window", "grey")
    # fbank
    plot_spectrum(fbank_feats, 5, 2, 8, "fbank", "grey")
    # mfcc
    plot_spectrum(mfcc_feats, 5, 2, 10, "mfcc", "grey")

    plt.show()

def main():
    wav, fs = librosa.load('./test.wav', sr=None)
    signal = preemphasis(wav)
    frames = enframe(signal)
    spectrum = get_spectrum(frames)
    fbank_feats = fbank(spectrum)
    mfcc_feats = mfcc(fbank_feats)
    plot_spectrogram(fbank_feats.T, 'Filter Bank','fbank.png')
    write_file(fbank_feats,'./test.fbank')
    plot_spectrogram(mfcc_feats.T, 'MFCC','mfcc.png')
    write_file(mfcc_feats,'./test.mfcc')

def test():
    mel_filter(23, 16000, 512)

if __name__ == '__main__':
    # main()
    test_every_step()
