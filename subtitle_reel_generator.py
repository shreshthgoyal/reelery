import os
import json
import pandas as pd
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, ImageClip, CompositeAudioClip, TextClip, ColorClip
import textwrap
import requests
from pathlib import Path
import re

# ensure IMv7+ uses "magick" binary
os.environ.setdefault("IMAGEMAGICK_BINARY", "magick")

class SubtitleReelGenerator:
    def __init__(self, config_file="config.json", timing_file="audio_timings.json"):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        self.video_config = self.config["video"]
        os.makedirs(self.video_config["output_folder"], exist_ok=True)
        
        self.timing_data = {}
        if os.path.exists(timing_file):
            with open(timing_file, 'r') as f:
                timing_list = json.load(f)
                self.timing_data = {item["reelid"]: item for item in timing_list}
            print(f"Loaded timing data for {len(self.timing_data)} reels")
        else:
            print(f"No timing file found at {timing_file}, will use CSV timings")
        
        # Remove speaker images loading since we only have male voice
        self.speaker_images = {}
        
        # use PIL-based text rendering with custom font
        self.subtitle_config = {
            "font": "Arial-Black",
            "fontsize": 85,  # Slightly smaller for better fit
            "text_color": "#FFFF00",  # Pure white for better contrast
            "highlight_color": "#FFD700",  # Gold highlight
            "stroke_color": "#000000",  # Black stroke for contrast
            "stroke_width": 6,  # Increased stroke for better visibility
            "method": "label",  # use PIL/TextClip label method
            "align": "center",    # center text without errors
            "position_y_ratio": 0.5,  # Moved slightly lower
            "position_x": 50,  # x offset from left edge
            "background_opacity": 0.0,  # No background
            "line_spacing": 1.2,
            "max_width_ratio": 0.95,  # Slightly more width usage
            "padding": 20,
            "shadow_color": (0, 0, 0, 120),  # Stronger shadow
            "shadow_offset": (3, 3),  # (dx, dy) for shadow
            "word_highlight_duration": 0.6,
            "smooth_transition": True,
            "fade_duration": 0.08,
            "letter_spacing": 1,
            "words_per_second": 2.5,
            "use_gradient": True,  # New gradient effect
            "outline_layers": 3  # Multiple outline layers for better visibility
        }

    def _parse_time(self, value, default):
        if pd.isna(value): return default
        if isinstance(value, (int, float)): return float(value)
        s = str(value).strip()
        if ":" in s:
            parts = s.split(":")
            try:
                if len(parts) == 2:
                    return int(parts[0]) * 60 + float(parts[1])
                if len(parts) == 3:
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            except:
                return default
        try:
            return float(s)
        except:
            return default

    def _get_segment_timing_data(self, reel_id, segment_type):
        """Get timing data for a specific segment"""
        if reel_id not in self.timing_data:
            return None
        
        segments = self.timing_data[reel_id].get("segments", [])
        for segment in segments:
            if segment["type"] == segment_type:
                return segment
        return None

    def _get_word_timings_for_segment(self, reel_id, segment_type):
        """Get word timings for a specific segment"""
        # First try to get from segment-level word_timings
        segment_data = self._get_segment_timing_data(reel_id, segment_type)
        if segment_data and 'word_timings' in segment_data and segment_data['word_timings']:
            word_timings = segment_data['word_timings']
            print(f"Found {len(word_timings)} word timings for {segment_type} (segment-level)")
            return word_timings
        
        # Fallback: extract from reel-level word timings
        if reel_id in self.timing_data and 'word_timings' in self.timing_data[reel_id]:
            all_word_timings = self.timing_data[reel_id]['word_timings']
            segment_word_timings = [
                wt for wt in all_word_timings 
                if wt.get('segment') == segment_type
            ]
            if segment_word_timings:
                print(f"Found {len(segment_word_timings)} word timings for {segment_type} (reel-level fallback)")
                return segment_word_timings
        else:
            print(f"No word timings found for {segment_type}")
            return None

    def _create_enhanced_text_clip(self, text, duration, video_size, start_time=0):
        """Create text clip with enhanced visibility and blending"""
        try:
            text_to_display = str(text)
            x = self.subtitle_config["position_x"]
            y = int(video_size[1] * self.subtitle_config["position_y_ratio"])
            dx, dy = self.subtitle_config["shadow_offset"]
            max_w = int(video_size[0] * self.subtitle_config["max_width_ratio"])
            
            # Shadow with automatic wrapping
            shadow = TextClip(
                text_to_display,
                fontsize=self.subtitle_config["fontsize"],
                font=self.subtitle_config["font"],
                color="black",
                method="caption", 
                size=(max_w, None),
                align=self.subtitle_config["align"]
            ).set_duration(duration).set_start(start_time)
            shadow = shadow.set_position((x+dx, y+dy)).set_opacity(0.6)
            
            # Main text with automatic wrapping
            main = TextClip(
                text_to_display,
                fontsize=self.subtitle_config["fontsize"],
                font=self.subtitle_config["font"],
                color=self.subtitle_config["text_color"],
                stroke_color=self.subtitle_config["stroke_color"],
                stroke_width=self.subtitle_config["stroke_width"],
                method="caption", 
                size=(max_w, None),
                align=self.subtitle_config["align"]
            ).set_duration(duration).set_start(start_time)
            main = main.set_position((x, y))
            
            return [shadow, main]
        except Exception as e:
            print(f"Error creating enhanced text clip: {e}")
            return []

    def _create_synchronized_word_subtitles(self, word_timings, video_size, speaker=None):
        """Create subtitles synchronized with precise word timings from TTS"""
        if not word_timings:
            print("No word timings provided to _create_synchronized_word_subtitles")
            return []
        
        print(f"Creating synchronized subtitles for {len(word_timings)} words")
        subtitle_clips = []
        words_per_screen = 6
        
        word_chunks = []
        i = 0
        while i < len(word_timings):
            chunk_size = 6 if (i // 6) % 2 == 0 else 7
            chunk = word_timings[i:i + chunk_size]
            word_chunks.append(chunk)
            i += chunk_size
        
        print(f"Created {len(word_chunks)} word chunks")
        
        for chunk_idx, chunk in enumerate(word_chunks):
            chunk_end = chunk[-1]['end']
            next_chunk_start = None
            
            if chunk_idx + 1 < len(word_chunks):
                next_chunk_start = word_chunks[chunk_idx + 1][0]['start']
            
            for word_idx_in_chunk in range(len(chunk)):
                current_word_timing = chunk[word_idx_in_chunk]
                displayed_words = chunk[:word_idx_in_chunk + 1]
                
                display_start = current_word_timing['start']
                
                if word_idx_in_chunk + 1 < len(chunk):
                    display_end = chunk[word_idx_in_chunk + 1]['start']
                else:
                    if next_chunk_start is not None:
                        display_end = next_chunk_start
                    else:
                        display_end = chunk_end
                
                display_duration = max(display_end - display_start, 0.1)
                
                try:
                    display_text = " ".join([wt['word'] for wt in displayed_words])
                    
                    # Don't wrap - use original text length for word animation
                    
                    # Create enhanced text clips
                    enhanced_clips = self._create_enhanced_text_clip(
                        display_text, display_duration, video_size, display_start
                    )
                    
                    # Apply fade effects - fix the fade application
                    is_first_word = (chunk_idx == 0 and word_idx_in_chunk == 0)
                    
                    for clip in enhanced_clips:
                        if is_first_word:
                            clip = clip.fadein(0.15)
                        subtitle_clips.append(clip)
                    
                except Exception as e:
                    print(f"Error creating subtitle for word '{current_word_timing['word']}': {e}")
                    continue
        
        print(f"Created {len(subtitle_clips)} subtitle clips total")
        return subtitle_clips

    def _create_simple_segment_subtitle(self, text, start_time, duration, video_size, speaker=None):
        try:
            text_to_display = str(text)
            x = self.subtitle_config["position_x"]
            y = int(video_size[1] * self.subtitle_config["position_y_ratio"])
            dx, dy = self.subtitle_config["shadow_offset"]
            max_w = int(video_size[0] * self.subtitle_config["max_width_ratio"])
            
            shadow = TextClip(
                text_to_display,
                fontsize=self.subtitle_config["fontsize"],
                font=self.subtitle_config["font"],
                color="black",
                method="caption",
                size=(max_w, None),
                align=self.subtitle_config["align"]
            ).set_duration(duration).set_start(start_time)
            shadow = shadow.set_position((x+dx, y+dy)).set_opacity(0.6).fadein(0.2)
            
            main = TextClip(
                text_to_display,
                fontsize=self.subtitle_config["fontsize"],
                font=self.subtitle_config["font"],
                color=self.subtitle_config["text_color"],
                stroke_color=self.subtitle_config["stroke_color"],
                stroke_width=self.subtitle_config["stroke_width"],
                method="caption",
                size=(max_w, None),
                align=self.subtitle_config["align"]
            ).set_duration(duration).set_start(start_time)
            main = main.set_position((x, y)).fadein(0.2)
            
            return [shadow, main]
        except Exception as e:
            print(f"Error creating simple subtitle: {e}")
            return []

    def _create_enhanced_subtitles(self, text, start_time, duration, video_size, segment_type=None, reel_id=None, speaker=None):
        print(f"_create_enhanced_subtitles called for segment_type: {segment_type}, reel_id: {reel_id}")
        print(f"Text: '{text}', Start: {start_time}, Duration: {duration}")
        
        if reel_id and segment_type:
            word_timings = self._get_word_timings_for_segment(reel_id, segment_type)
            if word_timings:
                print(f"Using word-level animation for {segment_type}")
                return self._create_synchronized_word_subtitles(word_timings, video_size)
        
        print(f"Falling back to simple subtitle for {segment_type}")
        # Create a simple fallback subtitle
        try:
            text_to_display = str(text)
            x = self.subtitle_config["position_x"]
            y = int(video_size[1] * self.subtitle_config["position_y_ratio"])
            dx, dy = self.subtitle_config["shadow_offset"]
            max_w = int(video_size[0] * self.subtitle_config["max_width_ratio"])
            
            # Shadow
            shadow = TextClip(
                text_to_display,
                fontsize=self.subtitle_config["fontsize"],
                font=self.subtitle_config["font"],
                color="black",
                method="caption",
                size=(max_w, None),
                align=self.subtitle_config["align"]
            ).set_duration(duration).set_start(start_time)
            shadow = shadow.set_position((x+dx, y+dy)).set_opacity(0.6).fadein(0.2)
            
            # Main
            main = TextClip(
                text_to_display,
                fontsize=self.subtitle_config["fontsize"],
                font=self.subtitle_config["font"],
                color=self.subtitle_config["text_color"],
                stroke_color=self.subtitle_config["stroke_color"],
                stroke_width=self.subtitle_config["stroke_width"],
                method="caption",
                size=(max_w, None),
                align=self.subtitle_config["align"]
            ).set_duration(duration).set_start(start_time)
            main = main.set_position((x, y)).fadein(0.2)
            
            print(f"Created fallback subtitle clips for '{text[:30]}...'")
            return [shadow, main]
            
        except Exception as e:
            print(f"Error creating fallback subtitle: {e}")
            return []

    def generate_reel(self, csv_file, reel_id=None, use_word_animation=True):
        try:
            df = pd.read_csv(csv_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
            
        if df.empty:
            raise ValueError(f"CSV file is empty: {csv_file}")

        if reel_id is not None:
            if reel_id not in df["reelid"].values:
                raise ValueError(f"Reel ID '{reel_id}' not found in CSV file")
            row = df[df["reelid"] == reel_id].iloc[0]
        else:
            row = df.iloc[0]
            reel_id = row["reelid"]
            
        print(f"Generating reel with conversation subtitles: {reel_id}")
        
        reel_timing_data = self.timing_data.get(reel_id)
        if reel_timing_data:
            target_duration = reel_timing_data.get("total_duration", 60) + 2
            print(f"Using TTS-calculated duration: {target_duration:.1f}s")
        else:
            cta_st = self._parse_time(row.get("cta_start"), 0)
            cta_du = self._parse_time(row.get("cta_duration"), 0)
            cta_end = cta_st + cta_du
            target_duration = cta_end + 1 if cta_end > 0 else 60
            print(f"Using CSV-based duration: {target_duration:.1f}s")
            
        dims = self.video_config["output_size"]
        
        video_folder = self.video_config["video_folder"]
        audio_folder = self.video_config["audio_folder"]
        if not os.path.exists(video_folder):
            raise FileNotFoundError(f"Video folder not found: {video_folder}")

        video = os.path.join(video_folder, f"bg{np.random.randint(1, 6)}.mp4")
        if not os.path.exists(video):
            for ext in (".mov", ".avi"): 
                p = video.replace(".mp4", ext)
                if os.path.exists(p): 
                    video = p
                    break
            else:
                templates = [f for f in os.listdir(video_folder) if f.endswith((".mp4", ".mov", ".avi"))]
                if not templates: 
                    raise FileNotFoundError(f"No template video found for {reel_id} in {video_folder}")
                video = os.path.join(video_folder, templates[0])
                
        print(f"Using background video: {video}")
        
        try:
            vc = VideoFileClip(video)
        except Exception as e:
            raise Exception(f"Error loading video file: {e}")
            
        vc = vc.loop(duration=target_duration) if vc.duration < target_duration else vc.subclip(0, target_duration)
        tw, th = dims
        vc = vc.resize(height=th)
        if vc.w > tw:
            cx = vc.w / 2
            vc = vc.crop(x1=cx-tw/2, y1=0, x2=cx+tw/2, y2=th)
        vc = vc.fadein(1).fadeout(1)
        vc = vc.without_audio()
        
        clips = [vc]
        tts_audio_clips = []
        subtitle_clips = []

        if reel_timing_data and reel_timing_data.get("segments"):
            for segment in reel_timing_data["segments"]:
                segment_type = segment["type"]
                text = segment["text"]
                start_time = segment["start"]
                duration = segment["duration"]
                
                print(f"Processing {segment_type}: {start_time:.1f}s - {start_time + duration:.1f}s ({duration:.1f}s)")
                
                audio_file = os.path.join(audio_folder, segment["audio_file"])
                if os.path.exists(audio_file):
                    audio_clip = AudioFileClip(audio_file).set_start(start_time)
                    audio_clip = audio_clip.audio_fadein(0.2).audio_fadeout(0.2)
                    tts_audio_clips.append(audio_clip)
                    print(f"  ✓ Added audio: {segment['audio_file']}")
                else:
                    print(f"  ⚠ Audio file not found: {audio_file}")
                
                # Remove speaker image logic
                
                if use_word_animation:
                    print(f"Creating word animation for {segment_type}")
                    segment_subtitles = self._create_enhanced_subtitles(
                        text, start_time, duration, dims, segment_type, reel_id
                    )
                else:
                    print(f"Creating simple subtitle for {segment_type}")
                    segment_subtitles = self._create_simple_segment_subtitle(
                        text, start_time, duration, dims
                    )
                
                subtitle_clips.extend(segment_subtitles)
                print(f"  ✓ Added {len(segment_subtitles)} subtitle clips")
        
        else:
            print("No TTS timing data available, using CSV-based timing")
            
            i = 1
            current_time = 0
            while f"segment{i}" in row:
                txt = row[f"segment{i}"]
                if pd.notna(txt) and txt:
                    seg_duration = 8
                    
                    seg_wav = os.path.join(audio_folder, f"{reel_id}_segment{i}.wav")
                    if os.path.exists(seg_wav):
                        audio_clip = AudioFileClip(seg_wav).set_start(current_time)
                        audio_clip = audio_clip.audio_fadein(0.15).audio_fadeout(0.15)
                        tts_audio_clips.append(audio_clip)
                    
                    seg_subtitles = self._create_enhanced_subtitles(
                        txt, current_time, seg_duration, dims, f"segment{i}", reel_id
                    )
                    subtitle_clips.extend(seg_subtitles)
                    
                    current_time += seg_duration
                i += 1
            
            if pd.notna(row.get("cta")):
                cta_start = current_time
                cta_duration = 5
                
                cta_wav = os.path.join(audio_folder, f"{reel_id}_cta.wav")
                if os.path.exists(cta_wav):
                    audio_clip = AudioFileClip(cta_wav).set_start(cta_start)
                    audio_clip = audio_clip.audio_fadein(0.2).audio_fadeout(0.2)
                    tts_audio_clips.append(audio_clip)
                
                cta_subtitles = self._create_enhanced_subtitles(
                    row["cta"], cta_start, cta_duration, dims, "cta", reel_id
                )
                subtitle_clips.extend(cta_subtitles)

        clips.extend(subtitle_clips)
        
        # Load background music
        bg_music_file = os.path.join(audio_folder, "bg.mp3")
        bg_music_clip = None
        if os.path.exists(bg_music_file):
            try:
                bg_music_clip = AudioFileClip(bg_music_file)
                # Loop background music to match video duration
                if bg_music_clip.duration < target_duration:
                    bg_music_clip = bg_music_clip.loop(duration=target_duration)
                else:
                    bg_music_clip = bg_music_clip.subclip(0, target_duration)
                # Reduce volume for background music (30% of original volume)
                bg_music_clip = bg_music_clip.volumex(0.3)
                print(f"✓ Added background music: bg.mp3 (volume: 30%)")
            except Exception as e:
                print(f"⚠ Error loading background music: {e}")
                bg_music_clip = None
        else:
            print("⚠ Background music file (bg.mp3) not found")

        # Composite all audio clips
        all_audio_clips = []
        if tts_audio_clips:
            all_audio_clips.extend(tts_audio_clips)
        if bg_music_clip:
            all_audio_clips.append(bg_music_clip)
            
        ac = CompositeAudioClip(all_audio_clips) if all_audio_clips else None

        final_duration = target_duration
        if ac and tts_audio_clips:
            end_times = [cl.start + cl.duration for cl in tts_audio_clips]
            max_audio_end = max(end_times) if end_times else final_duration
            final_duration = max(final_duration, max_audio_end + 1)

        final = CompositeVideoClip(clips, size=dims).set_audio(ac).set_duration(final_duration)
        out = os.path.join(self.video_config["output_folder"], f"{reel_id}.mp4")
        
        print(f"Writing video with {len(subtitle_clips)} synchronized subtitle clips to {out}")
        print(f"Final video duration: {final_duration:.1f}s")
        
        try:
            final.write_videofile(
                out,
                codec="libx264",
                audio_codec="aac",
                fps=30,
                audio=True,
                temp_audiofile="temp-audio.m4a",
                remove_temp=True,
                threads=4,
                preset="fast",                    # faster encode with near-lossless quality
                ffmpeg_params=["-crf", "18"],     # constant rate factor for quality
                verbose=False,
                logger=None
            )
        except Exception as e:
            raise Exception(f"Error writing video file: {e}")
        
        # Cleanup resources
        vc.close()
        if ac and hasattr(ac, 'close'): 
            ac.close()
        if bg_music_clip and hasattr(bg_music_clip, 'close'):
            bg_music_clip.close()
        final.close()
        
        print(f"✓ Successfully generated reel with conversation subtitles: {out}")

        os.remove(f"inputs/audio/{reel_id}_hook.wav")
        os.remove(f"inputs/audio/{reel_id}.wav")
        return out

    def set_subtitle_style(self, **kwargs):
        self.subtitle_config.update(kwargs)
        print(f"Updated subtitle config: {kwargs}")

    def get_timing_info(self, reel_id):
        if reel_id in self.timing_data:
            return self.timing_data[reel_id]
        return None

    def list_available_reels(self):
        return list(self.timing_data.keys())