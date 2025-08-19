from transformers import AutoProcessor, VitsModel
import torchaudio

# Model for multilingual + voice cloning
model_name = "coqui/XTTS-v2"
from transformers import pipeline

pipe = pipeline("text-to-speech", model=model_name)

# Input text with emotion hints
text = "Narrator, [happy and excited]: Today, we begin our magical journey through the forest."

# Your custom voice sample
speaker_wav = "custom_voice_sample.wav"

# Generate audio
audio = pipe(text, speaker_wav=speaker_wav, language="en")
with open("output.wav", "wb") as f:
    f.write(audio["audio"])
