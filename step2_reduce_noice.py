import noisereduce as nr
import soundfile as sf

INPUT_FILE = "books/sample/musical_prompt/test_10/narration_1.wav"
OUTPUT_FILE = "books/sample/musical_prompt/test_10/narration_1_cleaned.wav"

from pydub import AudioSegment, effects

# Load audio
audio = AudioSegment.from_wav(INPUT_FILE)

# Normalize loudness
normalized = effects.normalize(audio)

# Export cleaned audio
normalized.export(OUTPUT_FILE, format="wav")
