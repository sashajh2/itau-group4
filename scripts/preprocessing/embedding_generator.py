from retriever.embedders import AUDIO_EMBEDDERS, VIDEO_EMBEDDERS, DENOISERS
from retriever.embedders.hubert_embedder import HubertEmbedder
from retriever.embedders.openl3_embedder import Openl3Embedder
from moviepy import VideoFileClip
from utils.embedding_utils import get_audio_array
import numpy as np

def embed_segments(segments):
    """
    Generate embeddings for all segments using all available embedders and denoisers.
    
    Args:
        segments: List of segment dictionaries
        
    Returns:
        accumulator: Dictionary with (model, mode) keys containing embeddings and segment_ids
    """
    # Create accumulator with regular audio embedders
    accumulator = {
        (e.model_name, e.mode): {"embeddings": [], "segment_ids": []}
        for e in AUDIO_EMBEDDERS + VIDEO_EMBEDDERS
    }
    
    # Add entries for denoised/noise embedders that will be created dynamically
    for embedder in AUDIO_EMBEDDERS:
        if embedder.mode == "audio":
            for denoiser_name in DENOISERS.keys():
                # Add entries for both denoised and noise modes
                key_denoised = (f"{embedder.model_name}_{denoiser_name}", "audio_denoised")
                key_noise = (f"{embedder.model_name}_{denoiser_name}", "audio_noise")
                accumulator[key_denoised] = {"embeddings": [], "segment_ids": []}
                accumulator[key_noise] = {"embeddings": [], "segment_ids": []}

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
            audio_array = get_audio_array(audio, sr)

            # Process audio with denoisers to get noised and denoised versions
            denoised_audio = {}
            noise_audio = {}
            
            for denoiser_name, denoiser in DENOISERS.items():
                try:
                    # Use the new split_audio function for efficiency
                    if denoiser_name == "voicefixer":
                        denoised, noise = denoiser.split_audio(audio_array, sr=sr)
                    else:
                        denoised, noise = denoiser.split_audio(audio_array)
                    
                    denoised_audio[denoiser_name] = denoised
                    noise_audio[denoiser_name] = noise
                except Exception as e:
                    print(f"❌ Denoising fail with {denoiser_name} for {segment_id}: {e}")

            # Process regular audio embedders
            for embedder in AUDIO_EMBEDDERS:
                if embedder.mode == "audio":
                    try:
                        emb = embedder.embed(audio_array, sr)
                        key = (embedder.model_name, embedder.mode)
                        accumulator[key]["embeddings"].append(emb)
                        accumulator[key]["segment_ids"].append(segment_id)
                    except Exception as e:
                        print(f"❌ Audio embed fail {segment_id} with {embedder.model_name}: {e}")

            # Process denoised audio embedders
            for denoiser_name in DENOISERS.keys():
                if denoiser_name in denoised_audio:
                    # Create embedders for denoised audio
                    hubert_denoised = HubertEmbedder(mode="audio_denoised")
                    openl3_denoised = Openl3Embedder(mode="audio_denoised")
                    
                    for embedder in [hubert_denoised, openl3_denoised]:
                        try:
                            emb = embedder.embed(denoised_audio[denoiser_name], sr)
                            key = (f"{embedder.model_name}_{denoiser_name}", embedder.mode)
                            accumulator[key]["embeddings"].append(emb)
                            accumulator[key]["segment_ids"].append(segment_id)
                        except Exception as e:
                            print(f"❌ Denoised embed fail {segment_id} with {embedder.model_name}_{denoiser_name}: {e}")

            # Process noise audio embedders
            for denoiser_name in DENOISERS.keys():
                if denoiser_name in noise_audio:
                    # Create embedders for noise audio
                    hubert_noise = HubertEmbedder(mode="audio_noise")
                    openl3_noise = Openl3Embedder(mode="audio_noise")
                    
                    for embedder in [hubert_noise, openl3_noise]:
                        try:
                            emb = embedder.embed(noise_audio[denoiser_name], sr)
                            key = (f"{embedder.model_name}_{denoiser_name}", embedder.mode)
                            accumulator[key]["embeddings"].append(emb)
                            accumulator[key]["segment_ids"].append(segment_id)
                        except Exception as e:
                            print(f"❌ Noise embed fail {segment_id} with {embedder.model_name}_{denoiser_name}: {e}")

            # Process video embedders
            for embedder in VIDEO_EMBEDDERS:
                try:
                    temp_video = (
                        embedder.get_video_noise(video)
                        if embedder.mode == "video noise"
                        else video
                    )
                    emb = embedder.embed(temp_video)
                    key = (embedder.model_name, embedder.mode)
                    accumulator[key]["embeddings"].append(emb)
                    accumulator[key]["segment_ids"].append(segment_id)
                except Exception as e:
                    print(f"❌ Video embed fail {segment_id} with {embedder.model_name}: {e}")

        except Exception as e:
            print(f"❌ Segment load fail {segment_id}: {e}")

    return accumulator 