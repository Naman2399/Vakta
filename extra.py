# import torch
# from transformers import AutoTokenizer
# from parler_tts import ParlerTTSForConditionalGeneration
# import soundfile as sf
#
# # Load model
# model_name = "parler-tts/parler-tts-mini-expresso"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
#
# # Example voice descriptions
# voices = [
#     "young female voice with a cheerful tone",
#     "middle-aged male voice with deep tone",
#     "elderly female voice with warm tone",
#     "young male voice with energetic tone",
#     "soft female voice with calm tone",
# ]
#
# def generate_voice_sample(description, text, output_path):
#     desc_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
#     text_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
#
#     gen = model.generate(input_ids=desc_ids, prompt_input_ids=text_ids)
#     audio = gen.cpu().numpy().squeeze()
#     sf.write(output_path, audio, model.config.sampling_rate)
#     print(f"Generated: {output_path}")
#
# # Generate samples
# sample_text = "This is a short sample narration for testing voice style."
# for i, voice_desc in enumerate(voices, start=1):
#     generate_voice_sample(
#         description=voice_desc,
#         text=sample_text,
#         output_path=f"voice_sample_{i}.wav"
#     )


# musicgen_audiocraft_duration.py
# import torch
# from audiocraft.models import MusicGen
# from audiocraft.data.audio import audio_write  # utility to save with loudness normalization
# import torchaudio
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# # Choose model scale: "small", "medium", "large", or "melody"
# model = MusicGen.get_pretrained("small", device=device)   # change to medium/large if you have memory
# # Set generation params (duration in seconds)
# duration_sec = 15  # desired duration
# model.set_generation_params(duration=duration_sec, use_sampling=True, top_k=250)
#
# # Provide one or more text descriptions (list)
# descriptions = ["A calming piano and strings soundtrack with a gentle rhythm"]
#
# # Generate (returns list of torch tensors - one per description)
# wavs = model.generate(descriptions)
#
# # Save outputs
# for i, wav in enumerate(wavs):
#     fname = f"musicgen_output_{i}.wav"
#     # audio_write handles normalization and saving
#     audio_write(fname, wav.cpu(), model.sample_rate, strategy="loudness")
#     print("Saved", fname)


from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained("medium")
model.set_generation_params(duration=8)  # generate 8 seconds.

descriptions = ["happy rock", "energetic EDM", "drums and bass", "war drums", "swords clashing"]

wav = model.generate(descriptions)  # generates 2 samples.

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness")
