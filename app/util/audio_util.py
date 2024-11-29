import io, wave
import numpy as np

def calc_audio_volume(bytes_chunk):
    audio_data = np.frombuffer(bytes_chunk, dtype=np.int16)
    rms = np.sqrt(np.mean(audio_data**2))
    mean_db = 20 * np.log10(rms)
    peak_db = 20 * np.log10(np.max(np.abs(audio_data)))
    return float(mean_db), float(peak_db)

# from https://huggingface.co/spaces/coqui/voice-chat-with-mistral/blob/main/app.py
def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=16000):
    # This will create a wave header then append the frame input
    # It should be first on a streaming wav file
    # Other frames better should not have it (else you will hear some artifacts each chunk start)
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)
    wav_buf.seek(0)
    return wav_buf.read(-1)