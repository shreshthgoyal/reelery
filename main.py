import os
import argparse
import json
import pandas as pd
import glob
import time
from datetime import datetime, timedelta
from tts_generator import TTSGenerator
from subtitle_reel_generator import SubtitleReelGenerator
from text_preview_generator import TextPreviewGenerator
import requests

def download_font(font_path):
    font_name = os.path.basename(font_path)
    urls = {
        "Montserrat-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat-Bold.ttf",
        "Roboto-Regular.ttf":  "https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Regular.ttf",
        "Roboto-Bold.ttf": "https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Bold.ttf"
    }
    url = urls.get(font_name)
    if not url:
        print(f"No URL for font {font_name}, skipping download.")
        return
    response = requests.get(url)
    response.raise_for_status()
    os.makedirs(os.path.dirname(font_path), exist_ok=True)
    with open(font_path, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded {font_name} to {font_path}")

def cleanup_generated_files(reel_id, preserve_video_path=None, preserve_audio_path=None):
    """Clean up temporary files generated during reel creation, keeping only final outputs"""
    files_to_clean = []
    
    # Clean up individual TTS audio files (keep concatenated audio)
    audio_pattern = f"inputs/audio/*{reel_id}*.mp3"
    audio_files = glob.glob(audio_pattern)
    for audio_file in audio_files:
        # Keep the concatenated audio file if specified
        if preserve_audio_path and os.path.abspath(audio_file) == os.path.abspath(preserve_audio_path):
            continue
        files_to_clean.append(audio_file)
    
    # Clean up any temporary subtitle files
    subtitle_pattern = f"*{reel_id}*.srt"
    subtitle_files = glob.glob(subtitle_pattern)
    files_to_clean.extend(subtitle_files)
    
    # Clean up any temporary processing files in outputs directory
    temp_output_pattern = f"outputs/*{reel_id}*temp*"
    temp_files = glob.glob(temp_output_pattern)
    files_to_clean.extend(temp_files)
    
    # Clean up individual segment audio files
    segment_audio_patterns = [
        f"inputs/audio/{reel_id}_hook*.mp3",
        f"inputs/audio/{reel_id}_segment*.mp3", 
        f"inputs/audio/{reel_id}_cta*.mp3"
    ]
    for pattern in segment_audio_patterns:
        files_to_clean.extend(glob.glob(pattern))
    
    cleaned_count = 0
    for file_path in files_to_clean:
        try:
            if os.path.exists(file_path):
                # Double check we're not deleting the final video or concatenated audio
                if preserve_video_path and os.path.abspath(file_path) == os.path.abspath(preserve_video_path):
                    continue
                if preserve_audio_path and os.path.abspath(file_path) == os.path.abspath(preserve_audio_path):
                    continue
                    
                os.remove(file_path)
                cleaned_count += 1
                print(f"Cleaned up: {file_path}")
        except Exception as e:
            print(f"Warning: Could not clean up {file_path}: {e}")
    
    if cleaned_count > 0:
        print(f"✅ Cleaned up {cleaned_count} temporary files for reel {reel_id}")
    else:
        print(f"No temporary files found to clean up for reel {reel_id}")

def main():
    parser = argparse.ArgumentParser(description="AI Reel Generation Pipeline")
    parser.add_argument("--config", default="config.json", help="Path to configuration file")
    parser.add_argument("--csv", default="inputs/scripts/content.csv", help="Path to CSV file with reel data (defaults to config value)")
    parser.add_argument("--reelid", help="Generate specific reel by ID")
    parser.add_argument("--all", action="store_true", help="Process all rows in the CSV")
    parser.add_argument("--skip-tts", action="store_true", help="Skip TTS generation")
    parser.add_argument("--download-font", action="store_true", help="Download font if not available")
    # Add new preview options
    parser.add_argument("--preview", action="store_true", help="Generate text preview images instead of full video")
    parser.add_argument("--preview-type", choices=["all", "hook", "cta", "segment1", "segment2", "segment3"], 
                        default="all", help="Type of preview to generate")
    parser.add_argument("--composite", action="store_true", help="Generate a composite preview with all text elements")
    parser.add_argument("--only-tts", action="store_true", help="Generate only TTS audio files")
    
    args = parser.parse_args()
    
    # Start overall timing
    overall_start_time = time.time()
    print(f"🚀 Starting reel generation pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found.")
        return
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    os.makedirs("inputs/scripts", exist_ok=True)
    os.makedirs("inputs/audio", exist_ok=True)
    os.makedirs("inputs/videos", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/previews", exist_ok=True)
    
    font_path = config["video"].get("font_path")
    if font_path and not os.path.exists(font_path):
        try:
            download_font(font_path)
        except Exception as e:
            print(f"Warning: Could not download font: {e}")
    
    # ensure fonts
    for fp in [config["video"].get("hook_font_path"), 
               config["video"].get("segment_font_path"),
               config["video"].get("cta_font_path")]:
        if fp and not os.path.exists(fp):
            try:
                download_font(fp)
            except Exception as e:
                print(f"Warning: could not download font {fp}: {e}")
    
    csv_file = args.csv if args.csv else config["video"]["csv_file"]
    
    if not os.path.exists(csv_file):
        print(f"Error: CSV file {csv_file} not found.")
        return
    
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded CSV with {len(df)} reel entries")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
    
    if args.reelid:
        rows_to_process = df[df["reelid"] == args.reelid]
        if rows_to_process.empty:
            print(f"Error: Reel ID {args.reelid} not found in CSV.")
            return
    else:
        # process all reels by default
        rows_to_process = df
    
    successful_reels = 0
    failed_reels = 0
    total_tts_time = 0
    total_video_time = 0
    
    for _, row in rows_to_process.iterrows():
        reel_id = row["reelid"]
        reel_start_time = time.time()
        print(f"\n=== Processing Reel ID: {reel_id} ===")
        print(f"⏰ Started at: {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            temp_csv = f"inputs/scripts/temp_{reel_id}.csv"
            pd.DataFrame([row]).to_csv(temp_csv, index=False)
            
            if args.only_tts:
                if not args.skip_tts:
                    tts_start = time.time()
                    tts_generator = TTSGenerator(args.config)
                    audio_file = tts_generator.process_csv_script(temp_csv)
                    tts_time = time.time() - tts_start
                    total_tts_time += tts_time
                    print(f"⏱️ TTS Generation Time: {tts_time:.2f} seconds")
                    if not audio_file:
                        print(f"Warning: Failed to generate audio for reel {reel_id}")
                os.remove(temp_csv)
                continue

            if args.preview or args.composite:
                preview_start = time.time()
                print(f"\n=== Generating Text Preview for Reel ID: {reel_id} ===")
                preview_generator = TextPreviewGenerator(args.config)
                
                if args.composite:
                    output_path = preview_generator.preview_all_text_at_once(temp_csv, reel_id)
                    print(f"✅ Composite preview generated: {output_path}")
                    print(f"Preview saved to: {output_path}")
                else:
                    output_paths = preview_generator.generate_preview(
                        temp_csv, 
                        reel_id=reel_id, 
                        preview_type=args.preview_type
                    )
                    print(f"✅ Preview images generated: {len(output_paths)} images")
                    for path in output_paths:
                        print(f"Preview saved to: {path}")
                
                preview_time = time.time() - preview_start
                print(f"⏱️ Preview Generation Time: {preview_time:.2f} seconds")
                successful_reels += 1
            else:
                # Regular video generation workflow
                generated_audio_file = None
                tts_time = 0
                
                if not args.skip_tts:
                    tts_start = time.time()
                    print(f"\n=== Generating TTS Audio for Reel ID: {reel_id} ===")
                    tts_generator = TTSGenerator(args.config)
                    generated_audio_file = tts_generator.process_csv_script(temp_csv)
                    tts_time = time.time() - tts_start
                    total_tts_time += tts_time
                    print(f"⏱️ TTS Generation Time: {tts_time:.2f} seconds")
                    if not generated_audio_file:
                        print(f"Warning: Failed to generate audio for reel {reel_id}")
                
                video_start = time.time()
                print(f"\n=== Generating Video Reel for Reel ID: {reel_id} ===")
                subtitle_reel_generator = SubtitleReelGenerator(args.config)
                subtitle_reel_generator.set_subtitle_style(
                    fontsize=75,
                    stroke_width=3,
                    color="yellow",
                    stroke_color="black",
                    background_opacity=0,
                    position_y_ratio=0.5,
                    words_per_reveal=1,
                    fade_duration=0.1,
                )
                output_path = subtitle_reel_generator.generate_reel(temp_csv, reel_id)
                video_time = time.time() - video_start
                total_video_time += video_time
                print(f"⏱️ Video Generation Time: {video_time:.2f} seconds")
                
                if output_path:
                    print(f"✅ Reel generated successfully: {output_path}")
                    successful_reels += 1
                    
                    cleanup_start = time.time()
                    print(f"\n=== Cleaning up temporary files for Reel ID: {reel_id} ===")
                    cleanup_generated_files(reel_id, preserve_video_path=output_path, preserve_audio_path=generated_audio_file)
                    cleanup_time = time.time() - cleanup_start
                    print(f"⏱️ Cleanup Time: {cleanup_time:.2f} seconds")
                else:
                    print(f"❌ Failed to generate reel for ID: {reel_id}")
                    failed_reels += 1
            
            reel_total_time = time.time() - reel_start_time
            print(f"⏱️ Total Reel Processing Time: {reel_total_time:.2f} seconds")
            print(f"✅ Completed at: {datetime.now().strftime('%H:%M:%S')}")
            
            os.remove(temp_csv)
        except Exception as e:
            reel_total_time = time.time() - reel_start_time
            print(f"Error processing reel {reel_id}: {e}")
            print(f"⏱️ Failed after: {reel_total_time:.2f} seconds")
            failed_reels += 1
    
    # Calculate overall timing
    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time
    
    print(f"\n=== Reel Generation Summary ===")
    print(f"🏁 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️ Total Pipeline Duration: {total_duration:.2f} seconds ({timedelta(seconds=int(total_duration))})")
    
    if total_tts_time > 0:
        print(f"🎵 Total TTS Time: {total_tts_time:.2f} seconds")
    if total_video_time > 0:
        print(f"🎬 Total Video Generation Time: {total_video_time:.2f} seconds")
    
    print(f"📊 Performance Stats:")
    print(f"   Total processed: {len(rows_to_process)}")
    print(f"   Successful: {successful_reels}")
    print(f"   Failed: {failed_reels}")
    
    if successful_reels > 0:
        avg_time_per_reel = total_duration / len(rows_to_process)
        print(f"   Average time per reel: {avg_time_per_reel:.2f} seconds")
        
        if total_video_time > 0:
            avg_video_time = total_video_time / successful_reels
            print(f"   Average video generation time: {avg_video_time:.2f} seconds")
        
        if total_tts_time > 0:
            avg_tts_time = total_tts_time / successful_reels
            print(f"   Average TTS generation time: {avg_tts_time:.2f} seconds")

if __name__ == "__main__":
    main()