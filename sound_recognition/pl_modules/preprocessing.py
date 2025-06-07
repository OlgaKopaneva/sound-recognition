import librosa
import numpy as np


def auto_complete_conf(conf):
    conf.samples = conf.sampling_rate * conf.duration
    conf.dims = (
        conf.n_mels,
        1 + conf.samples // conf.hop_length,
        1,
    )
    return conf


def read_audio(conf, pathname):
    y, _ = librosa.load(pathname, sr=conf.sampling_rate)
    if len(y) > 0:
        y, _ = librosa.effects.trim(y)
    if len(y) > conf.samples:
        if conf.audio_split == "head":
            y = y[: conf.samples]
    else:
        padding = conf.samples - len(y)
        offset = padding // 2
        y = np.pad(y, (offset, conf.samples - len(y) - offset), "constant")
    return y


def audio_to_melspectrogram(conf, audio):
    mel = librosa.feature.melspectrogram(
        audio=audio,
        sr=conf.sampling_rate,
        n_fft=conf.n_fft,
        hop_length=conf.hop_length,
        n_mels=conf.n_mels,
        fmin=conf.fmin,
        fmax=conf.fmax,
    )
    return librosa.power_to_db(mel).astype(np.float32)


def read_as_melspectrogram(conf, pathname):
    x = read_audio(conf, pathname)
    return audio_to_melspectrogram(conf, x)


def split_long_data(conf, melspec):
    L = melspec.shape[1]
    one_len = conf.dims[1]
    step = int(one_len * 0.9)
    min_len = int(one_len * 0.2)

    for idx in range(L // step):
        cur = step * idx
        if one_len <= L - cur:
            yield melspec[:, cur : cur + one_len]
        elif min_len <= L - cur:
            cur = L - one_len
            yield melspec[:, cur : cur + one_len]


def convert_X(df, conf, audio_dir):
    X, idx_map = [], []
    for i, fname in enumerate(df.fname):
        mels = read_as_melspectrogram(conf, audio_dir / fname)
        for chunk in split_long_data(conf, mels):
            X.append(np.expand_dims(chunk, axis=-1))
            idx_map.append(i)
    return np.array(X), np.array(idx_map)


def convert_y_train(idx_map, y):
    return np.array([y[i] for i in idx_map])
