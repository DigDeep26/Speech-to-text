import whisper
import sounddevice as sd
import numpy as np
import torch

# Load the Whisper model
model = whisper.load_model("base")  # You can use "tiny" for faster performance

# Audio configuration
SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
CHANNELS = 1         # Mono audio

def record_audio(duration=2):
    """Record audio for the given duration and return a NumPy array."""
    print(f"🎙️ Recording for {duration} seconds...")
    try:
        audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
        sd.wait()
        return audio.flatten()
    except Exception as e:
        print("❌ Error during recording:", e)
        return np.zeros(int(duration * SAMPLE_RATE), dtype='float32')

def transcribe_audio(audio):
    """Normalize and transcribe audio using Whisper."""
    try:
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        result = model.transcribe(audio, fp16=torch.cuda.is_available())
        return result.get("text", "")
    except Exception as e:
        print("❌ Error during transcription:", e)
        return ""

def main():
    print("🔊 Speak something... (say 'stop' to quit)")
    while True:
        audio = record_audio(duration=2)
        text = transcribe_audio(audio)
        print("📝 You said:", text)
        if "stop" in text.lower():
            print("🛑 Stopping transcription.")
            break

if __name__ == "__main__":
    main()
