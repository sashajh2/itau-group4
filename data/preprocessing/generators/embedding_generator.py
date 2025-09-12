from retriever.embedders import AUDIO_EMBEDDERS, VIDEO_EMBEDDERS, DENOISERS
from moviepy import VideoFileClip
from utils.embedding_utils import get_audio_array
import numpy as np
from collections import defaultdict

MODE_AUDIO = "audio"
MODE_VIDEO = "video"

NOISE_NONE = "none"
NOISE_DENOISED = "denoised"
NOISE_NOISY = "noisy"

def _embed_audio_with_sr(embedder, wav, sr):
    """Hubert expects 16k; others use the given SR."""
    if embedder.model_name == "hubert":
        # If caller didn't already give 16k, resample
        if sr != 16000:
            # assumes you have a helper; otherwise resample here
            wav16 = get_audio_array(wav, 16000)  # or your resampler
            return embedder.embed(wav16, 16000)
        return embedder.embed(wav, 16000)
    else:
        return embedder.embed(wav, sr)
        
def embed_segments(segments):
    """
    Generate embeddings for all segments using all available embedders and denoisers.
    
    Args:
        segments: List of segment dictionaries
        
    Returns:
        accumulator: Dictionary with (mode, model, noise, denoiser_name) keys containing embeddings and segment_ids
    """
    accumulator = defaultdict(lambda: {"embeddings": [], "segment_ids": []})

    for e in AUDIO_EMBEDDERS:
        for noise in (NOISE_NONE, NOISE_DENOISED, NOISE_NOISY):
            _ = accumulator[(MODE_AUDIO, e.model_name, noise, "none")]
    for e in VIDEO_EMBEDDERS:
        _ = accumulator[(MODE_VIDEO, e.model_name, NOISE_NONE, "none")]

    for i, segment in enumerate(segments):
        if i % 100 == 0:
            print(f"Processing segment {i+1}/{len(segments)}: {segment['segment_id']}")
            
        segment_id = segment["segment_id"]
        filepath = segment["video_path"]
        start_time = segment["start_time"]
        duration = segment["duration"]

        try:
            video = VideoFileClip(filepath)
            video = video.subclipped(start_time, start_time + duration)
            audio = video.audio
            sr = audio.fps
            wav = get_audio_array(audio, sr)
            
            # Create 16kHz audio array specifically for Hubert (which expects 16kHz)
            wav_16k = get_audio_array(audio, 16000)
            
            ### BASE AUDIO
            for embedder in AUDIO_EMBEDDERS:
                try:
                    # parameters for embedding
                    wav_to_embed = wav if embedder.model_name != "hubert" else wav_16k
                    sr_to_embed = sr if embedder.model_name != "hubert" else 16000

                    # generate embedding
                    emb = _embed_audio_with_sr(embedder, wav_to_embed, sr_to_embed)

                    # add to accumulator
                    key = (MODE_AUDIO, embedder.model_name, NOISE_NONE, "none")
                    accumulator[key]["embeddings"].append(emb)
                    accumulator[key]["segment_ids"].append(segment_id)
                except Exception as e:
                    print(f"❌ Audio embed fail {segment_id} with {embedder.model_name}: {e}")
        
            ### GENERATE DENOISED/NOISY AUDIO
            denoised_by_name = {}
            noisy_by_name    = {}
            denoised16_by_name = {}
            noisy16_by_name    = {}

            for denoiser_name, denoiser in DENOISERS.items():
                try:
                    # Support denoisers that want explicit sr
                    if denoiser_name == "voicefixer":
                        deno, noise = denoiser.split_audio(wav, sr=sr)
                        deno16, noise16 = denoiser.split_audio(wav_16k, sr=16000)
                    else:
                        deno, noise = denoiser.split_audio(wav)
                        deno16, noise16 = denoiser.split_audio(wav_16k)

                    denoised_by_name[denoiser_name]  = deno
                    noisy_by_name[denoiser_name]     = noise
                    denoised16_by_name[denoiser_name] = deno16
                    noisy16_by_name[denoiser_name]    = noise16
                except Exception as e:
                    print(f"❌ Denoising fail with {denoiser_name} for {segment_id}: {e}")

            ### Embed denoised/noisy audio
            for denoiser_name in DENOISERS.keys():
                if denoiser_name in denoised_by_name:
                    for embedder in AUDIO_EMBEDDERS:
                        try:
                            # parameters for embedding
                            wav_deno = denoised16_by_name[denoiser_name] if embedder.model_name == "hubert" else denoised_by_name[denoiser_name]
                            sr_deno  = 16000 if embedder.model_name == "hubert" else sr

                            # generate embedding
                            emb = _embed_audio_with_sr(embedder, wav_deno, sr_deno)

                            # add to accumulator
                            key = (MODE_AUDIO, embedder.model_name, NOISE_DENOISED, denoiser_name)
                            accumulator[key]["embeddings"].append(emb)
                            accumulator[key]["segment_ids"].append(segment_id)
                        except Exception as e:
                            print(f"❌ Denoised embed fail {segment_id} with {embedder.model_name}/{denoiser_name}: {e}")

                if denoiser_name in noisy_by_name:
                    for embedder in AUDIO_EMBEDDERS:
                        try:
                            # parameters for embedding
                            wav_noisy = noisy16_by_name[denoiser_name] if embedder.model_name == "hubert" else noisy_by_name[denoiser_name]
                            sr_noisy  = 16000 if embedder.model_name == "hubert" else sr

                            # generate embedding
                            emb = _embed_audio_with_sr(embedder, wav_noisy, sr_noisy)

                            # add to accumulator
                            key = (MODE_AUDIO, embedder.model_name, NOISE_NOISY, denoiser_name)
                            accumulator[key]["embeddings"].append(emb)
                            accumulator[key]["segment_ids"].append(segment_id)
                        except Exception as e:
                            print(f"❌ Noisy embed fail {segment_id} with {embedder.model_name}/{denoiser_name}: {e}")

            ### VIDEO
            for embedder in VIDEO_EMBEDDERS:
                try:
                    # parameters for embedding
                    emb = embedder.embed(video)
                    key = (MODE_VIDEO, embedder.model_name, NOISE_NONE, "none")
                    accumulator[key]["embeddings"].append(emb)
                    accumulator[key]["segment_ids"].append(segment_id)
                except Exception as e:
                    print(f"❌ Video embed fail {segment_id} with {embedder.model_name}: {e}")

        except Exception as e:
            print(f"❌ Segment load fail {segment_id}: {e}")

    return accumulator