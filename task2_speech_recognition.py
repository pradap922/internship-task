"""
TASK 2: SPEECH RECOGNITION SYSTEM
Transcribes short audio clips using pre-trained models.
Libraries: SpeechRecognition, pydub, (optional) wav2vec2 via HuggingFace
"""


import os
import wave
import struct
import math



def transcribe_with_speech_recognition(audio_path: str, engine: str = "google") -> str:
    """
    engine options: 'google' (requires internet) | 'sphinx' (offline, pip install pocketsphinx)
    """
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()

        with sr.AudioFile(audio_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)

        if engine == "google":
            text = recognizer.recognize_google(audio_data)
        elif engine == "sphinx":
            text = recognizer.recognize_sphinx(audio_data)
        else:
            raise ValueError(f"Unknown engine: {engine}")

        return text

    except ImportError:
        return "❌  Install SpeechRecognition: pip install SpeechRecognition"
    except Exception as e:
        return f"❌  Recognition error: {e}"



def transcribe_with_wav2vec2(audio_path: str) -> str:
    """Use facebook/wav2vec2-base-960h for offline transcription."""
    try:
        import torch
        import torchaudio
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

        tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        model     = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample to 16 kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform  = resampler(waveform)

        input_values = tokenizer(waveform.squeeze().numpy(),
                                 return_tensors="pt",
                                 sampling_rate=16000).input_values

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = tokenizer.decode(predicted_ids[0])
        return transcription

    except ImportError:
        return "❌  Install torch, torchaudio, transformers: pip install transformers torch torchaudio"
    except Exception as e:
        return f"❌  Wav2Vec2 error: {e}"



def record_from_microphone(output_path: str = "recorded.wav",
                            duration: int = 5,
                            sample_rate: int = 16000) -> str:
    """Record audio from the default microphone for `duration` seconds."""
    try:
        import pyaudio
        CHUNK  = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1

        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS,
                        rate=sample_rate, input=True,
                        frames_per_buffer=CHUNK)

        print(f"🎙️  Recording for {duration} seconds … speak now!")
        frames = [stream.read(CHUNK) for _ in range(int(sample_rate / CHUNK * duration))]
        print("⏹️  Recording stopped.")

        stream.stop_stream()
        stream.close()
        p.terminate()

        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))

        return output_path

    except ImportError:
        return "❌  Install pyaudio: pip install pyaudio"
    except Exception as e:
        return f"❌  Microphone error: {e}"



def _create_test_wav(path: str = "test_audio.wav"):
    """Creates a 1-second 440 Hz sine-wave WAV for testing."""
    sample_rate = 16000
    frequency   = 440
    duration    = 1
    amplitude   = 32767

    n_samples = sample_rate * duration
    samples   = [int(amplitude * math.sin(2 * math.pi * frequency * t / sample_rate))
                 for t in range(n_samples)]

    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(sample_rate)
        packed = struct.pack(f'<{n_samples}h', *samples)
        wf.writeframes(packed)
    return path



if __name__ == "__main__":
    print("=" * 60)
    print("       TASK 2 — SPEECH RECOGNITION SYSTEM")
    print("=" * 60)

    AUDIO_FILE = "your_audio.wav"   # ← replace with your WAV path

    if not os.path.exists(AUDIO_FILE):
        print(f"\n⚠️  '{AUDIO_FILE}' not found — generating a synthetic test tone …")
        AUDIO_FILE = _create_test_wav("test_audio.wav")
        print(f"✅  Created '{AUDIO_FILE}' (440 Hz sine wave — no meaningful speech).")

    print(f"\n📂  Audio file : {AUDIO_FILE}")

    print("\n🔍  METHOD 1 — SpeechRecognition (Google API)")
    print("-" * 60)
    result1 = transcribe_with_speech_recognition(AUDIO_FILE, engine="google")
    print(f"Transcript : {result1}")
    
    print("\n🔍  METHOD 2 — Wav2Vec2 (offline, HuggingFace)")
    print("-" * 60)
    result2 = transcribe_with_wav2vec2(AUDIO_FILE)
    print(f"Transcript : {result2}")

    print("\n\n🎙️  OPTION B — Live microphone recording")
    print("-" * 60)
    answer = input("Record from microphone? (y/n): ").strip().lower()
    if answer == 'y':
        mic_file = record_from_microphone("mic_recording.wav", duration=5)
        if mic_file.endswith(".wav"):
            print("\n🔍  Transcribing recording …")
            result3 = transcribe_with_speech_recognition(mic_file, engine="google")
            print(f"Transcript : {result3}")
    else:
        print("Skipped microphone recording.")

    print("\n" + "=" * 60)
    print("✅  Speech recognition demo complete!")
    print("=" * 60)
