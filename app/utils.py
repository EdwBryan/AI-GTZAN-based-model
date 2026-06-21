import warnings
import numpy as np
import librosa as li

AUDIO_EXTENSIONS = {".wav", ".mp3", ".au", ".flac", ".ogg", ".m4a"}

def extract_features_from_audio(y, sr=22050):
    mfcc = li.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_delta = li.feature.delta(mfcc)
    mfcc_delta2 = li.feature.delta(mfcc, order=2)

    chroma_stft = li.feature.chroma_stft(y=y, sr=sr)
    try:
        chroma_cqt = li.feature.chroma_cqt(y=y, sr=sr)
    except Exception:
        chroma_cqt = chroma_stft
    try:
        chroma_vqt = li.feature.chroma_vqt(y=y, sr=sr, intervals="12TET")
    except Exception:
        chroma_vqt = chroma_cqt

    spectral_centroid = li.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = li.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = li.feature.spectral_rolloff(y=y, sr=sr)
    spectral_contrast = li.feature.spectral_contrast(y=y, sr=sr)
    spectral_flatness = li.feature.spectral_flatness(y=y)

    tonnetz = li.feature.tonnetz(y=li.effects.harmonic(y), sr=sr)
    onset_env = li.onset.onset_strength(y=y, sr=sr)
    tempogram = li.feature.tempogram(onset_envelope=onset_env, sr=sr)

    rms = li.feature.rms(y=y)
    zcr = li.feature.zero_crossing_rate(y)
    mel_spec = li.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    tempo, _ = li.beat.beat_track(y=y, sr=sr)

    features = {
        "mfcc_mean": float(np.mean(mfcc)),
        "mfcc_std": float(np.std(mfcc)),
        "mfcc_delta_mean": float(np.mean(mfcc_delta)),
        "mfcc_delta_std": float(np.std(mfcc_delta)),
        "mfcc_delta2_mean": float(np.mean(mfcc_delta2)),
        "mfcc_delta2_std": float(np.std(mfcc_delta2)),
        "chroma_stft_mean": float(np.mean(chroma_stft)),
        "chroma_stft_std": float(np.std(chroma_stft)),
        "chroma_cqt_mean": float(np.mean(chroma_cqt)),
        "chroma_cqt_std": float(np.std(chroma_cqt)),
        "chroma_vqt_mean": float(np.mean(chroma_vqt)),
        "chroma_vqt_std": float(np.std(chroma_vqt)),
        "spectral_centroid_mean": float(np.mean(spectral_centroid)),
        "spectral_centroid_std": float(np.std(spectral_centroid)),
        "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
        "spectral_bandwidth_std": float(np.std(spectral_bandwidth)),
        "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
        "spectral_rolloff_std": float(np.std(spectral_rolloff)),
        "spectral_contrast_mean": float(np.mean(spectral_contrast)),
        "spectral_contrast_std": float(np.std(spectral_contrast)),
        "spectral_flatness_mean": float(np.mean(spectral_flatness)),
        "spectral_flatness_std": float(np.std(spectral_flatness)),
        "tonnetz_mean": float(np.mean(tonnetz)),
        "tonnetz_std": float(np.std(tonnetz)),
        "tempogram_mean": float(np.mean(tempogram)),
        "tempogram_std": float(np.std(tempogram)),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "zero_crossing_rate_mean": float(np.mean(zcr)),
        "zero_crossing_rate_std": float(np.std(zcr)),
        "mel_spec_mean": float(np.mean(mel_spec)),
        "mel_spec_std": float(np.std(mel_spec)),
        "tempo": float(np.atleast_1d(tempo).flat[0]),
    }
    return features


def load_audio(file_bytes, sr=22050):
    import soundfile as sf
    import io
    try:
        y, sr = li.load(io.BytesIO(file_bytes), sr=sr, mono=True)
        return y, sr
    except Exception:
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(io.BytesIO(file_bytes))
            wav_bytes = io.BytesIO()
            audio.export(wav_bytes, format="wav")
            wav_bytes.seek(0)
            y, sr = li.load(wav_bytes, sr=sr, mono=True)
            return y, sr
        except Exception as e:
            raise RuntimeError(f"Could not load audio: {e}")
