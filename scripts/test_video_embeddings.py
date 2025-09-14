#!/usr/bin/env python3
"""
Generate embeddings for test videos by chopping them into 0.25-second clips.

This script:
1. Takes videos from data/test_videos/
2. Chops each video into 0.25-second segments
3. Generates OpenL3, SENET, and Hubert embeddings for each segment
4. Saves embeddings as .npy files in embeddings/test/
"""

import os
import numpy as np
from moviepy import VideoFileClip
from tqdm import tqdm
import argparse

# Import the embedders
from retriever.embedders.openl3_embedder import Openl3Embedder
from retriever.embedders.senet_embedder import SenetEmbedder
from retriever.embedders.hubert_embedder import HubertEmbedder

# Constants
SEGMENT_DURATION = 0.25  # 0.25 seconds
TEST_VIDEOS_DIR = "data/test_videos"
OUTPUT_DIR = "embeddings/test"

def get_audio_array(audio, sr):
    """Convert moviepy audio to numpy array."""
    if audio is None:
        return None
    return np.array(audio.to_soundarray())

def embed_audio_with_sr(embedder, wav, sr):
    """Generate audio embedding with specific sample rate."""
    if wav is None:
        return None
    return embedder.embed(wav, sr)

def process_video(video_path, output_dir):
    """
    Process a single video file and generate embeddings for all segments.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save embeddings
        
    Returns:
        dict: Statistics about the processing
    """
    print(f"\n🎬 Processing video: {os.path.basename(video_path)}")
    
    # Initialize embedders
    openl3_embedder = Openl3Embedder()
    senet_embedder = SenetEmbedder()
    hubert_embedder = HubertEmbedder()
    
    # Load video
    try:
        video = VideoFileClip(video_path)
        total_duration = video.duration
        print(f"   Duration: {total_duration:.2f} seconds")
    except Exception as e:
        print(f"❌ Failed to load video: {e}")
        return {"error": str(e)}
    
    # Calculate number of segments
    num_segments = int(total_duration / SEGMENT_DURATION)
    print(f"   Will create {num_segments} segments of {SEGMENT_DURATION}s each")
    
    # Initialize embedding arrays
    openl3_embeddings = []
    senet_embeddings = []
    hubert_embeddings = []
    segment_info = []
    
    # Process each segment
    for i in tqdm(range(num_segments), desc="Processing segments", leave=False):
        start_time = i * SEGMENT_DURATION
        end_time = min((i + 1) * SEGMENT_DURATION, total_duration)
        
        try:
            # Extract segment
            segment_video = video.subclipped(start_time, end_time)
            audio = segment_video.audio
            
            if audio is None:
                print(f"⚠️  No audio for segment {i}")
                continue
                
            # Get audio array
            sr = audio.fps
            wav = get_audio_array(audio, sr)
            wav_16k = get_audio_array(audio, 16000)  # For Hubert
            
            if wav is None:
                print(f"⚠️  Failed to extract audio for segment {i}")
                continue
            
            # Generate OpenL3 embedding
            try:
                openl3_emb = embed_audio_with_sr(openl3_embedder, wav, sr)
                if openl3_emb is not None:
                    openl3_embeddings.append(openl3_emb)
                else:
                    print(f"⚠️  OpenL3 embedding failed for segment {i}")
            except Exception as e:
                print(f"❌ OpenL3 error for segment {i}: {e}")
            
            # Generate SENET embedding
            try:
                senet_emb = senet_embedder.embed(segment_video)
                if senet_emb is not None:
                    senet_embeddings.append(senet_emb)
                else:
                    print(f"⚠️  SENET embedding failed for segment {i}")
            except Exception as e:
                print(f"❌ SENET error for segment {i}: {e}")
            
            # Generate Hubert embedding
            try:
                hubert_emb = embed_audio_with_sr(hubert_embedder, wav_16k, 16000)
                if hubert_emb is not None:
                    hubert_embeddings.append(hubert_emb)
                else:
                    print(f"⚠️  Hubert embedding failed for segment {i}")
            except Exception as e:
                print(f"❌ Hubert error for segment {i}: {e}")
            
            # Store segment info
            segment_info.append({
                "segment_id": i,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time
            })
            
        except Exception as e:
            print(f"❌ Error processing segment {i}: {e}")
            continue
    
    # Close video
    video.close()
    
    # Save embeddings
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save OpenL3 embeddings
    if openl3_embeddings:
        openl3_path = os.path.join(output_dir, f"{video_name}_openl3_embeddings.npy")
        np.save(openl3_path, np.array(openl3_embeddings))
        print(f"✅ Saved OpenL3 embeddings: {openl3_path} ({len(openl3_embeddings)} segments)")
    else:
        print("⚠️  No OpenL3 embeddings generated")
    
    # Save SENET embeddings
    if senet_embeddings:
        senet_path = os.path.join(output_dir, f"{video_name}_senet_embeddings.npy")
        np.save(senet_path, np.array(senet_embeddings))
        print(f"✅ Saved SENET embeddings: {senet_path} ({len(senet_embeddings)} segments)")
    else:
        print("⚠️  No SENET embeddings generated")
    
    # Save Hubert embeddings
    if hubert_embeddings:
        hubert_path = os.path.join(output_dir, f"{video_name}_hubert_embeddings.npy")
        np.save(hubert_path, np.array(hubert_embeddings))
        print(f"✅ Saved Hubert embeddings: {hubert_path} ({len(hubert_embeddings)} segments)")
    else:
        print("⚠️  No Hubert embeddings generated")
    
    # Save segment info
    segment_info_path = os.path.join(output_dir, f"{video_name}_segment_info.npy")
    np.save(segment_info_path, segment_info)
    print(f"✅ Saved segment info: {segment_info_path}")
    
    return {
        "video_name": video_name,
        "total_duration": total_duration,
        "num_segments": num_segments,
        "openl3_count": len(openl3_embeddings),
        "senet_count": len(senet_embeddings),
        "hubert_count": len(hubert_embeddings)
    }

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for test videos")
    parser.add_argument("--input-dir", type=str, default=TEST_VIDEOS_DIR, help="Directory containing test videos")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Directory to save embeddings")
    parser.add_argument("--segment-duration", type=float, default=SEGMENT_DURATION, help="Duration of each segment in seconds")
    
    args = parser.parse_args()
    
    # Use argument values
    input_dir = args.input_dir
    output_dir = args.output_dir
    segment_duration = args.segment_duration
    
    print(f"🎬 Test Video Embedding Generator")
    print(f"   Input directory: {input_dir}")
    print(f"   Output directory: {output_dir}")
    print(f"   Segment duration: {segment_duration}s")
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"❌ Input directory does not exist: {input_dir}")
        return
    
    # Find video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    video_files = []
    
    for file in os.listdir(input_dir):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(os.path.join(input_dir, file))
    
    if not video_files:
        print(f"❌ No video files found in {input_dir}")
        return
    
    print(f"📁 Found {len(video_files)} video files:")
    for video_file in video_files:
        print(f"   - {os.path.basename(video_file)}")
    
    # Process each video
    results = []
    for video_file in video_files:
        result = process_video(video_file, output_dir)
        results.append(result)
    
    # Print summary
    print(f"\n📊 Processing Summary:")
    print(f"   Videos processed: {len(results)}")
    
    for result in results:
        if "error" in result:
            print(f"   ❌ {result['video_name']}: {result['error']}")
        else:
            print(f"   ✅ {result['video_name']}:")
            print(f"      Duration: {result['total_duration']:.2f}s")
            print(f"      Segments: {result['num_segments']}")
            print(f"      OpenL3: {result['openl3_count']} embeddings")
            print(f"      SENET: {result['senet_count']} embeddings")
            print(f"      Hubert: {result['hubert_count']} embeddings")
    
    print(f"\n🎉 Processing complete! Check {output_dir} for results.")

if __name__ == "__main__":
    main()
