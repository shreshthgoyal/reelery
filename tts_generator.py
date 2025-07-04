import os
import json
import pandas as pd
import torchaudio
import torch
import numpy as np
import edge_tts
import librosa
import re
import asyncio
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

class TTSGenerator:
    def __init__(self, config_file="config.json"):
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {}
        
        os.makedirs("inputs/audio", exist_ok=True)
        
        self.male_voice = "en-US-GuyNeural"
    
    async def generate_audio_async(self, text, output_path, voice=None):
        if not text or pd.isna(text):
            print(f"Warning: Empty text provided for {output_path}, skipping.")
            return None
            
        if voice is None:
            voice = self.male_voice
            
        print(f"Generating TTS with {voice} for: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        try:
            mp3_path = output_path.replace(".wav", ".mp3")
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(mp3_path)
            
            # load and trim silence
            waveform, sr = torchaudio.load(mp3_path)
            wav_np = waveform.numpy()
            trimmed_np, _ = librosa.effects.trim(wav_np, top_db=20)
            trimmed = torch.from_numpy(trimmed_np)
            torchaudio.save(output_path, trimmed, sr)
            os.remove(mp3_path)

            print(f"WAV audio saved to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error generating audio for '{text[:20]}...': {e}")
            return None

    def generate_audio(self, text, output_path, voice=None):
        return asyncio.run(self.generate_audio_async(text, output_path, voice))

    def get_audio_duration(self, audio_path):
        try:
            waveform, sr = torchaudio.load(audio_path)
            duration = waveform.shape[1] / sr
            return duration
        except Exception as e:
            print(f"Error getting duration for {audio_path}: {e}")
            return 0

    def get_audio_info(self, audio_path):
        try:
            waveform, sr = torchaudio.load(audio_path)
            duration = waveform.shape[1] / sr
            channels = waveform.shape[0]
            return {
                'duration': duration,
                'sample_rate': sr,
                'channels': channels,
                'samples': waveform.shape[1]
            }
        except Exception as e:
            print(f"Error getting audio info for {audio_path}: {e}")
            return None

    def _extract_unified_word_timings(self, segments_data):
        """
        Generate unified word timings across all segments for seamless subtitle display
        """
        unified_timings = []
        
        for segment_idx, segment in enumerate(segments_data):
            words = re.findall(r'\S+', segment['text'])
            if not words:
                continue
            
            segment_start = segment['start']
            segment_duration = segment['duration']
            
            # Calculate word timings within this segment
            word_weights = []
            for word in words:
                base_weight = len(word)
                syllable_count = max(1, self._estimate_syllables(word))
                complexity_factor = 1 + (syllable_count - 1) * 0.3
                
                # Punctuation adds pause time
                punctuation_factor = 1.0
                if any(p in word for p in '.,!?;:'):
                    punctuation_factor = 1.4
                
                # Common words are spoken faster
                if word.lower() in ['a', 'an', 'the', 'of', 'to', 'in', 'on', 'at', 'by', 'for', 'is', 'are', 'was', 'were']:
                    base_weight *= 0.7
                
                final_weight = base_weight * complexity_factor * punctuation_factor
                word_weights.append(final_weight)
            
            total_weight = sum(word_weights)
            if total_weight == 0:
                continue
            
            # Allocate 90% for speech, 10% for inter-word pauses
            speech_time = segment_duration * 0.9
            pause_time = segment_duration * 0.1
            inter_word_pause = pause_time / max(1, len(words) - 1) if len(words) > 1 else 0
            
            current_time = segment_start
            
            for i, (word, weight) in enumerate(zip(words, word_weights)):
                word_duration = (weight / total_weight) * speech_time
                
                unified_timings.append({
                    'word': word,
                    'start': float(current_time),
                    'end': float(current_time + word_duration),
                    'duration': float(word_duration),
                    'segment': segment['type']
                })
                
                current_time += word_duration
                
                # Add inter-word pause (except for last word)
                if i < len(words) - 1:
                    current_time += inter_word_pause
            
            # Ensure we don't exceed segment boundary
            if unified_timings and unified_timings[-1]['end'] > segment_start + segment_duration:
                excess = unified_timings[-1]['end'] - (segment_start + segment_duration)
                # Proportionally reduce all word timings in this segment
                segment_words = [t for t in unified_timings if t['segment'] == segment['type']]
                if segment_words:
                    scale_factor = (segment_duration - inter_word_pause * (len(segment_words) - 1)) / sum(t['duration'] for t in segment_words)
                    current_time = segment_start
                    for timing in segment_words:
                        timing['duration'] *= scale_factor
                        timing['start'] = current_time
                        timing['end'] = current_time + timing['duration']
                        current_time = timing['end'] + (inter_word_pause if timing != segment_words[-1] else 0)
        
        return unified_timings

    def _generate_subtitle_phrases(self, unified_timings, max_words_per_phrase=7, max_duration=4.0):
        """
        Group words into subtitle phrases for better readability
        """
        if not unified_timings:
            return []
        
        phrases = []
        current_phrase = []
        phrase_start = unified_timings[0]['start']
        
        for i, timing in enumerate(unified_timings):
            current_phrase.append(timing)
            
            # Check if we should end current phrase
            should_end_phrase = (
                len(current_phrase) >= max_words_per_phrase or  # Max words reached (increased to 7)
                (timing['end'] - phrase_start) >= max_duration or  # Max duration reached (increased to 4.0)
                timing['word'].endswith(('.', '!', '?')) or  # Sentence end
                (i < len(unified_timings) - 1 and 
                 unified_timings[i + 1]['start'] - timing['end'] > 0.5) or  # Long pause
                (i < len(unified_timings) - 1 and 
                 unified_timings[i + 1]['segment'] != timing['segment'])  # Segment change
            )
            
            if should_end_phrase or i == len(unified_timings) - 1:
                if current_phrase:
                    phrase_text = ' '.join([t['word'] for t in current_phrase])
                    phrases.append({
                        'text': phrase_text,
                        'start': float(phrase_start),
                        'end': float(current_phrase[-1]['end']),
                        'duration': float(current_phrase[-1]['end'] - phrase_start),
                        'word_count': len(current_phrase),
                        'words': current_phrase.copy()
                    })
                    
                    # Start new phrase
                    if i < len(unified_timings) - 1:
                        phrase_start = unified_timings[i + 1]['start']
                    current_phrase = []
        
        return phrases

    def _estimate_syllables(self, word):
        word = word.lower().strip('.,!?;:')
        if not word:
            return 1
        
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)

    def parse_time(self, value, default=None):
        if pd.isna(value):
            return default
            
        if isinstance(value, (int, float)):
            return float(value)
            
        if isinstance(value, str):
            value = value.strip()
            if ":" in value:
                try:
                    parts = value.split(":")
                    if len(parts) == 2:
                        minutes, seconds = parts
                        return int(minutes) * 60 + float(seconds)
                    elif len(parts) == 3:
                        hours, minutes, seconds = parts
                        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                except Exception:
                    return default
            try:
                return float(value)
            except Exception:
                return default
                
        return default

    def split_into_conversation(self, text):
        # Remove conversation splitting - just return single male voice segment
        return [{
            'text': text.strip(),
            'speaker': 'male',
            'part': 1
        }]

    def _parse_tagged_conversation(self, text):
        # Remove tag parsing - treat all text as male voice
        return [{
            'text': re.sub(r'\[(MALE|FEMALE)\]', '', text).strip(),
            'speaker': 'male', 
            'part': 1
        }]

    def process_csv_script(self, csv_file):
        try:
            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.strip()
        except Exception as e:
            print(f"Error reading CSV file {csv_file}: {e}")
            return None
            
        if df.empty:
            print(f"Error: CSV file {csv_file} is empty")
            return None
            
        results = []
        timing_data = []
        
        for index, row in df.iterrows():
            try:
                reel_id = row["reelid"]
                print(f"\nProcessing TTS conversation for reel ID: {reel_id}")
                
                audio_segments = []
                segments_data = []
                reel_timing = {
                    "reelid": reel_id,
                    "segments": [],
                    "word_timings": [],
                    "subtitle_phrases": [],
                    "metadata": {
                        "created_at": pd.Timestamp.now().isoformat(),
                        "total_words": 0,
                        "total_characters": 0,
                        "conversation_parts": 0
                    }
                }
                current_time = 0.0
                
                # Process hook/conversation - always use male voice
                if "hook" in row and pd.notna(row["hook"]) and str(row["hook"]).strip():
                    hook_text = str(row["hook"]).strip()
                    
                    voice = self.male_voice
                    segment_type = "hook"
                    audio_path = os.path.join("inputs/audio", f"{reel_id}_{segment_type}.wav")
                    
                    print(f"  Processing {segment_type} (male voice)...")
                    if self.generate_audio(hook_text, audio_path, voice):
                        audio_info = self.get_audio_info(audio_path)
                        if audio_info:
                            duration = audio_info['duration']
                            
                            segment_info = {
                                "type": segment_type,
                                "text": hook_text,
                                "voice": voice,
                                "start": float(current_time),
                                "end": float(current_time + duration),
                                "duration": float(duration),
                                "audio_file": os.path.basename(audio_path),
                                "word_count": len([w for w in re.findall(r'\S+', hook_text)]),
                                "character_count": len(hook_text),
                                "audio_info": audio_info
                            }
                            
                            audio_segments.append((audio_path, current_time, duration))
                            segments_data.append(segment_info)
                            reel_timing["segments"].append(segment_info)
                            
                            reel_timing["metadata"]["total_words"] += segment_info["word_count"]
                            reel_timing["metadata"]["total_characters"] += segment_info["character_count"]
                            
                            current_time += duration
                            print(f"    ✓ Generated {segment_type}: {duration:.2f}s")
                        else:
                            print(f"    ✗ Failed to get audio info for {segment_type}")
                    else:
                        print(f"    ✗ Failed to generate audio for {segment_type}")
                
                # Process other segments
                segment_types = ["cta"]
                
                i = 1
                while f"segment{i}" in row:
                    segment_types.insert(-1, f"segment{i}")
                    i += 1
                
                for segment_type in segment_types:
                    if segment_type in row and pd.notna(row[segment_type]) and str(row[segment_type]).strip():
                        text = str(row[segment_type]).strip()
                        audio_path = os.path.join("inputs/audio", f"{reel_id}_{segment_type}.wav")
                        
                        print(f"  Processing {segment_type}...")
                        if self.generate_audio(text, audio_path):
                            audio_info = self.get_audio_info(audio_path)
                            if audio_info:
                                duration = audio_info['duration']
                                
                                segment_info = {
                                    "type": segment_type,
                                    "text": text,
                                    "start": float(current_time),
                                    "end": float(current_time + duration),
                                    "duration": float(duration),
                                    "audio_file": os.path.basename(audio_path),
                                    "word_count": len([w for w in re.findall(r'\S+', text)]),
                                    "character_count": len(text),
                                    "audio_info": audio_info
                                }
                                
                                audio_segments.append((audio_path, current_time, duration))
                                segments_data.append(segment_info)
                                reel_timing["segments"].append(segment_info)
                                
                                reel_timing["metadata"]["total_words"] += segment_info["word_count"]
                                reel_timing["metadata"]["total_characters"] += segment_info["character_count"]
                                
                                current_time += duration
                                print(f"    ✓ Generated {segment_type}: {duration:.2f}s")
                            else:
                                print(f"    ✗ Failed to get audio info for {segment_type}")
                        else:
                            print(f"    ✗ Failed to generate audio for {segment_type}")
                
                reel_timing["metadata"]["conversation_parts"] = 1  # Always 1 since no conversation
                
                # Generate unified word timings for the entire reel
                print(f"  Generating unified subtitle timings...")
                unified_word_timings = self._extract_unified_word_timings(segments_data)
                reel_timing["word_timings"] = unified_word_timings
                
                # Distribute word timings to their respective segments
                print(f"  Distributing word timings to segments...")
                for segment in reel_timing["segments"]:
                    segment_timings = [
                        wt for wt in unified_word_timings 
                        if wt["segment"] == segment["type"]
                    ]
                    segment["word_timings"] = segment_timings
                    print(f"    Added {len(segment_timings)} word timings to {segment['type']}")
                
                # Generate subtitle phrases
                subtitle_phrases = self._generate_subtitle_phrases(unified_word_timings)
                reel_timing["subtitle_phrases"] = subtitle_phrases
                
                reel_timing["total_duration"] = float(current_time)
                reel_timing["metadata"]["segment_count"] = len(reel_timing["segments"])
                reel_timing["metadata"]["phrase_count"] = len(subtitle_phrases)
                reel_timing["metadata"]["average_segment_duration"] = float(current_time / len(reel_timing["segments"])) if reel_timing["segments"] else 0
                
                timing_data.append(reel_timing)
                
                # Combine audio segments
                if audio_segments:
                    combined_audio_path = os.path.join("inputs/audio", f"{reel_id}.wav")
                    if self._combine_audio_segments(audio_segments, combined_audio_path):
                        print(f"✓ Created combined audio file: {combined_audio_path}")
                        results.append(combined_audio_path)
                        
                        combined_duration = self.get_audio_duration(combined_audio_path)
                        if abs(combined_duration - current_time) > 0.5:
                            print(f"Warning: Combined audio duration ({combined_duration:.2f}s) doesn't match expected ({current_time:.2f}s)")
                    else:
                        print(f"✗ Failed to create combined audio for reel ID {reel_id}")
                else:
                    print(f"✗ No audio segments generated for reel ID {reel_id}")
                    
            except Exception as e:
                print(f"Error processing reel ID {row.get('reelid', 'unknown')}: {e}")
                import traceback
                traceback.print_exc()
        
        # Save timing data
        timing_file = "audio_timings.json"
        with open(timing_file, 'w') as f:
            json.dump(timing_data, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Audio timing data saved to {timing_file}")
        
        # Generate reports
        # self._generate_timing_report(timing_data)
        # self._generate_subtitle_files(timing_data)
        
        return results, timing_data

    def _generate_subtitle_files(self, timing_data):
        """Generate SRT subtitle files for each reel"""
        try:
            for reel in timing_data:
                reel_id = reel['reelid']
                srt_file = f"subtitles_{reel_id}.srt"
                
                with open(srt_file, 'w', encoding='utf-8') as f:
                    for i, phrase in enumerate(reel['subtitle_phrases'], 1):
                        start_time = self._seconds_to_srt_time(phrase['start'])
                        end_time = self._seconds_to_srt_time(phrase['end'])
                        
                        f.write(f"{i}\n")
                        f.write(f"{start_time} --> {end_time}\n")
                        f.write(f"{phrase['text']}\n\n")
                
                print(f"✓ Subtitle file saved: {srt_file}")
                
        except Exception as e:
            print(f"Error generating subtitle files: {e}")

    def _seconds_to_srt_time(self, seconds):
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    def _generate_timing_report(self, timing_data):
        try:
            report_file = "timing_report.txt"
            with open(report_file, 'w') as f:
                f.write("UNIFIED SUBTITLE TIMING REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                for reel in timing_data:
                    f.write(f"Reel ID: {reel['reelid']}\n")
                    f.write(f"Total Duration: {reel['total_duration']:.2f}s\n")
                    f.write(f"Segments: {reel['metadata']['segment_count']}\n")
                    f.write(f"Subtitle Phrases: {reel['metadata']['phrase_count']}\n")
                    f.write(f"Total Words: {reel['metadata']['total_words']}\n\n")
                    
                    f.write("SUBTITLE PHRASES:\n")
                    f.write("-" * 20 + "\n")
                    for i, phrase in enumerate(reel['subtitle_phrases'], 1):
                        f.write(f"  {i:2d}. {phrase['start']:6.2f}s - {phrase['end']:6.2f}s: {phrase['text']}\n")
                    
                    f.write("\nSEGMENT BREAKDOWN:\n")
                    f.write("-" * 20 + "\n")
                    for segment in reel['segments']:
                        speaker_info = f" ({segment.get('speaker', 'N/A')})" if 'speaker' in segment else ""
                        f.write(f"  {segment['type'].upper()}{speaker_info}: {segment['start']:.2f}s - {segment['end']:.2f}s\n")
                        f.write(f"    Text: {segment['text'][:80]}{'...' if len(segment['text']) > 80 else ''}\n")
                    
                    f.write("\n" + "=" * 50 + "\n\n")
                    
            print(f"✓ Timing report saved to {report_file}")
        except Exception as e:
            print(f"Error generating timing report: {e}")

    def _combine_audio_segments(self, audio_segments, output_path):
        if not audio_segments:
            print("Error: No audio segments to combine")
            return False
            
        try:
            target_sr = 24000
            target_channels = 1
            
            # Get audio properties from first valid segment
            valid_segments = []
            for path, start, duration in audio_segments:
                if os.path.exists(path):
                    try:
                        waveform, sr = torchaudio.load(path)
                        target_sr = sr
                        target_channels = waveform.shape[0]
                        valid_segments.append((path, start, duration))
                        break
                    except Exception as e:
                        print(f"Warning: Could not load {path}: {e}")
                        continue
            
            if not valid_segments:
                print("Error: No valid audio segments found")
                return False
            
            # Add remaining valid segments
            for path, start, duration in audio_segments:
                if (path, start, duration) not in valid_segments and os.path.exists(path):
                    try:
                        torchaudio.load(path)
                        valid_segments.append((path, start, duration))
                    except Exception as e:
                        print(f"Warning: Skipping invalid segment {path}: {e}")
            
            valid_segments.sort(key=lambda x: x[1])
            
            # Calculate total duration and create output buffer
            max_end_time = max(start + duration for _, start, duration in valid_segments)
            total_samples = int((max_end_time + 1) * target_sr)
            
            output_buffer = torch.zeros(target_channels, total_samples)
            
            # Combine all segments
            for path, start, duration in valid_segments:
                try:
                    waveform, sr = torchaudio.load(path)
                    
                    if sr != target_sr:
                        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
                    
                    if waveform.shape[0] != target_channels:
                        if waveform.shape[0] == 1 and target_channels == 2:
                            waveform = waveform.repeat(2, 1)
                        elif waveform.shape[0] == 2 and target_channels == 1:
                            waveform = torch.mean(waveform, dim=0, keepdim=True)
                    
                    start_sample = int(start * target_sr)
                    end_sample = min(start_sample + waveform.shape[1], total_samples)
                    segment_samples = end_sample - start_sample
                    
                    if segment_samples > 0:
                        output_buffer[:, start_sample:end_sample] += waveform[:, :segment_samples]
                    
                except Exception as e:
                    print(f"Error processing segment {path}: {e}")
                    continue
            
            # Normalize audio
            max_val = torch.max(torch.abs(output_buffer))
            if max_val > 1.0:
                output_buffer = output_buffer / max_val * 0.95
            
            # Trim silence at end
            non_zero_samples = torch.nonzero(torch.abs(output_buffer) > 0.001)
            if len(non_zero_samples) > 0:
                last_sample = non_zero_samples[-1, 1].item()
                output_buffer = output_buffer[:, :last_sample + int(0.5 * target_sr)]
            
            torchaudio.save(output_path, output_buffer, target_sr)
            
            # Verify saved file
            try:
                test_waveform, test_sr = torchaudio.load(output_path)
                actual_duration = test_waveform.shape[1] / test_sr
                print(f"✓ Combined audio saved: {actual_duration:.2f}s duration")
                return True
            except Exception as e:
                print(f"Error verifying combined audio: {e}")
                return False
            
        except Exception as e:
            print(f"Error combining audio segments: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_from_csv(self, csv_file):
        return self.process_csv_script(csv_file)