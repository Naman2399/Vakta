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


# from audiocraft.models import MusicGen
# from audiocraft.data.audio import audio_write
#
# model = MusicGen.get_pretrained("medium")
# model.set_generation_params(duration=8)  # generate 8 seconds.
#
# descriptions = ["happy rock", "energetic EDM", "drums and bass", "war drums", "swords clashing"]
#
# wav = model.generate(descriptions)  # generates 2 samples.
#
# for idx, one_wav in enumerate(wav):
#     # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
#     audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness")


# from bark import SAMPLE_RATE, generate_audio, preload_models
# from scipy.io.wavfile import write as write_wav
# from IPython.display import Audio
#
# # download and load all models
# preload_models()
#
# # generate audio from text
# text_prompt = """
#      Hello, my name is Suno. And, uh — and I like pizza. [laughs]
#      But I also have other interests such as playing tic tac toe.
# """
# audio_array = generate_audio(text_prompt)
#
# # save audio to disk
# write_wav("bark_generation.wav", SAMPLE_RATE, audio_array)
#
# # play text in notebook
# Audio(audio_array, rate=SAMPLE_RATE)


# import torchaudio
# from audiocraft.models import MusicGen
#
# # Load the pre-trained MusicGen model
# model = MusicGen.get_pretrained('facebook/musicgen-large')  # use 'small' if low on VRAM
# # model = MusicGen.get_pretrained('facebook/musicgen-melody')
#
# model.set_generation_params(duration=8)  # 8 seconds per sample
#
# # List of 10 prompts
# prompts = [
#     "energetic orchestral background music",
#     "calm ambient background with soft piano",
#     "dark suspenseful cinematic score",
#     "bright uplifting acoustic guitar music",
#     "fast-paced techno beat for action scene",
#     "emotional strings with a slow tempo",
#     "happy upbeat ukulele with claps",
#     "epic fantasy soundtrack with choir",
#     "jazzy lounge music with smooth saxophone",
#     "relaxing ocean waves with soft synth pads"
# ]
#
# # Generate audio for all prompts
# wavs = model.generate(prompts)
#
# # Save each result
# for i, wav in enumerate(wavs):
#     filename = f"music_sample_{i+1}.wav"
#     torchaudio.save(filename, wav.cpu(), 32000)
#     print(f"Saved: {filename}")


# import torchaudio
# from audiocraft.models import MusicGen
#
# # Load model on GPU with half precision for speed
# model = MusicGen.get_pretrained('facebook/musicgen-large', device='cuda')  # use 'small' for even faster
# # model = model.to("cuda").half()
#
# # Set shorter duration for faster output
# model.set_generation_params(duration=8)  # seconds
#
# # List of prompts
# prompts = [
#     "energetic orchestral background music",
#     "calm ambient background with soft piano",
#     "dark suspenseful cinematic score",
#     "bright uplifting acoustic guitar music",
#     "fast-paced techno beat for action scene",
#     "emotional strings with a slow tempo",
#     "happy upbeat ukulele with claps",
#     "epic fantasy soundtrack with choir",
#     "jazzy lounge music with smooth saxophone",
#     "relaxing ocean waves with soft synth pads"
# ]
#
# # Generate and save each prompt individually
# for prompt in prompts:
#     print(f"Generating: {prompt}")
#     wav = model.generate([prompt])
#     filename = f"{prompt[:20].replace(' ', '_')}.wav"
#     torchaudio.save(filename, wav[0].cpu(), 32000)
#     print(f"Saved: {filename}")


# import numpy as np
# import soundfile as sf
# import librosa
#
# def postprocess_wav(input_path, output_path, fade_len=200, trim_db=30):
#     # Load audio
#     wav, sr = sf.read(input_path)
#     wav = wav.astype(np.float32)
#
#     # Trim leading/trailing silence (librosa energy-based)
#     wav, _ = librosa.effects.trim(wav, top_db=trim_db)
#
#     # Apply fade-in
#     fade_in = np.linspace(0, 1, fade_len)
#     wav[:fade_len] *= fade_in
#
#     # Apply fade-out
#     fade_out = np.linspace(1, 0, fade_len)
#     wav[-fade_len:] *= fade_out
#
#     # Normalize loudness
#     wav = wav / (np.max(np.abs(wav)) + 1e-8)
#
#     # Save cleaned file
#     sf.write(output_path, wav, sr)
#     print(f"✅ Postprocessed file saved: {output_path}")
#
#     return wav, sr
#
# final_wav, sr = postprocess_wav("books/sample/musical_prompt/test_10/narration_1.wav", "books/sample/musical_prompt/test_10/narration_1_cleaned.wav")


# import re
#
# import numpy as np
# import soundfile as sf
# from TTS.api import TTS
#
#
# # --------- Split Text into Paragraph-Sized Chunks ----------
# def split_into_paragraphs(text, max_len=250):
#     sentences = re.split(r'(?<=[।.!?])\s+', text.strip())
#     sentences = [s.strip() for s in sentences if s.strip()]
#
#     chunks, current = [], ""
#     for s in sentences:
#         if len(current) + len(s) < max_len:
#             current += " " + s
#         else:
#             chunks.append(current.strip())
#             current = s
#     if current:
#         chunks.append(current.strip())
#     return chunks
#
#
# # --------- Concatenate with Small Silence Padding ----------
# def concat_with_silence(audios, sr, silence_sec=0.2):
#     silence = np.zeros(int(silence_sec * sr), dtype=np.float32)
#     final = []
#     for idx, wav in enumerate(audios):
#         final.append(wav)
#         if idx < len(audios) - 1:
#             final.append(silence)
#     return np.concatenate(final)
#
#
# # --------- Apply Fade-In/Fade-Out ----------
# def apply_fades(wav, fade_len=200):
#     fade_in = np.linspace(0, 1, fade_len)
#     wav[:fade_len] *= fade_in
#     fade_out = np.linspace(1, 0, fade_len)
#     wav[-fade_len:] *= fade_out
#     return wav
#
#
# # --------- Denoise with Torchaudio ----------
# def denoise_audio(wav: np.ndarray, sr: int) -> np.ndarray:
#     """
#     Denoise audio using spectral gating (noisereduce).
#     - Estimates noise profile from the quietest 0.5 sec.
#     - Works with any sample rate.
#     """
#
#     try:
#         import noisereduce as nr
#
#         # take first 0.5s (or less if audio shorter) as noise profile
#         noise_len = min(len(wav), sr // 2)
#         noise_clip = wav[:noise_len]
#
#         reduced = nr.reduce_noise(
#             y=wav,
#             sr=sr,
#             y_noise=noise_clip,
#             stationary=False,  # adaptive noise profile
#             prop_decrease=1.0  # aggressiveness, can tune (0.8–1.0)
#         )
#
#         return reduced.astype(np.float32)
#
#     except ImportError:
#         print("[WARN] noisereduce not installed, returning raw audio.")
#         return wav.astype(np.float32)
# # --------- Main TTS Pipeline ----------
# def generate_tts(
#     text,
#     reference_wav,
#     language="hi",
#     device="cuda",
#     output_path="final_output.wav"
# ):
#     # Load TTS model
#     tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
#
#     # Split text
#     paragraphs = split_into_paragraphs(text)
#     print(f"Processing {len(paragraphs)} chunks...")
#
#     all_audio = []
#     for idx, para in enumerate(paragraphs, 1):
#         wav = tts.tts(
#             text=para,
#             speaker_wav=reference_wav,
#             language=language
#         )
#         wav = np.array(wav, dtype=np.float32)
#
#         # Normalize loudness
#         wav = wav / (np.max(np.abs(wav)) + 1e-8)
#
#         all_audio.append(wav)
#         print(f"✅ Generated chunk {idx}/{len(paragraphs)}")
#
#     # Concatenate with silence padding
#     sr = 24000
#     combined = concat_with_silence(all_audio, sr, silence_sec=0.2)
#
#     # Apply fade smoothing
#     combined = apply_fades(combined, fade_len=400)
#
#     # Save intermediate raw file
#     sf.write("raw_output.wav", combined, sr)
#     print("💾 Raw file saved: raw_output.wav")
#
#     # Denoise
#     clean = denoise_audio(combined, sr)
#
#     # Apply final fade-out
#     clean = apply_fades(clean, fade_len=400)
#
#     # Save final output
#     sf.write(output_path, clean, sr)
#     print(f"🎧 Final clean audio saved: {output_path}")
#
#
# # --------- Example Usage ----------
# if __name__ == "__main__":
#     sample_text =  "In a world where the soft rustle of paper pages told tales, a time existed when stories were captured in the delicate folds of yellowed sheets. Tommy and Margie found themselves reminiscing about these relics from the past, laughing at the peculiar notion of words remaining still, while their screens danced with colors and movement. As they shared a moment together, their imaginations wandered back to that bygone era, where every turn of the page revealed the same cherished words they had read before, timeless and awaiting their eager eyes, a concept foreign to their vibrant, ever-changing displays."
#
#     reference_wav = "books/sample/HarishBhimaniVoiceSample.wav"  # <-- your clean reference voice sample
#     generate_tts(
#         text=sample_text,
#         reference_wav=reference_wav,
#         language="en",
#         device="cuda",
#         output_path="final_clean.wav"
#     )



# import torch
# print("Torch version:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())
# print("CUDA version:", torch.version.cuda)
# print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
#
#
# import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_THREADING_LAYER"] = "GNU"
#
# from audiocraft.models import MusicGen
# torch.cuda.empty_cache()
# print("Loading model...")
# model = MusicGen.get_pretrained("facebook/musicgen-melody", device="cuda")
# print("Model loaded OK")

import torch
from diffusers import CosmosTextToWorldPipeline
from diffusers.utils import export_to_video

model_id = "nvidia/Cosmos-1.0-Diffusion-7B-Text2World"
pipe = CosmosTextToWorldPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "A sleek, humanoid robot stands in a vast warehouse filled with neatly stacked cardboard boxes on industrial shelves. The robot's metallic body gleams under the bright, even lighting, highlighting its futuristic design and intricate joints. A glowing blue light emanates from its chest, adding a touch of advanced technology. The background is dominated by rows of boxes, suggesting a highly organized storage system. The floor is lined with wooden pallets, enhancing the industrial setting. The camera remains static, capturing the robot's poised stance amidst the orderly environment, with a shallow depth of field that keeps the focus on the robot while subtly blurring the background for a cinematic effect."

output = pipe(prompt=prompt).frames[0]
export_to_video(output, "output.mp4", fps=30)

