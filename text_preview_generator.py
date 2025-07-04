import os
import json
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
import textwrap
import requests
from pathlib import Path
import io
import cv2
import sys


class TextPreviewGenerator:
    def __init__(self, config_file="config.json"):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        self.video_config = self.config["video"]
        os.makedirs(self.video_config["output_folder"], exist_ok=True)
        
        self.preview_dir = Path(self.video_config["output_folder"]) / "previews"
        self.preview_dir.mkdir(exist_ok=True)
        
        self.fonts_dir = Path("fonts")
        self.fonts_dir.mkdir(exist_ok=True)
        
        vc = self.video_config
        self.hook_font_path = vc.get("hook_font_path", str(self.fonts_dir / "Bebas-Neue.ttf"))
        self.segment_font_path = vc.get("segment_font_path", str(self.fonts_dir / "Bebas-Neue.ttf"))
        self.cta_font_path = vc.get("cta_font_path", str(self.fonts_dir / "Bebas-Neue.ttf"))
        
        self.font_urls = {
    "hook": "https://fonts.gstatic.com/s/bebasneue/v14/JTUSjIg69CK48gW7PXooxW5rygbi49c.ttf",
    "segment": "https://fonts.gstatic.com/s/bebasneue/v14/JTUSjIg69CK48gW7PXooxW5rygbi49c.ttf",
    "cta": "https://fonts.gstatic.com/s/bebasneue/v14/JTUSjIg69CK48gW7PXooxW5rygbi49c.ttf"
}


    def _parse_time(self, value, default):
        if pd.isna(value):
            return default
        if isinstance(value, (int, float)):
            return float(value)
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

    def _ensure_font(self, font_path, font_type):
        if os.path.exists(font_path):
            return font_path
            
        url = self.font_urls.get(font_type)
        print(url)
        if not url:
            print(f"Warning: No URL configured for font type '{font_type}'")
            return None
            
        os.makedirs(os.path.dirname(font_path), exist_ok=True)
        
        try:
            print(f"Downloading font from {url}")
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            with open(font_path, 'wb') as f:
                f.write(r.content)
            print(f"Font successfully downloaded to {font_path}")
            return font_path
        except requests.exceptions.RequestException as e:
            print(f"Error downloading font: {e}")
            return None
        except Exception as e:
            print(f"Error saving font: {e}")
            return None

    def _load_font(self, font_path, font_type, font_size):
        if font_path and os.path.exists(font_path):
            try:
                print(f"Found local font: {font_path}")
                return ImageFont.truetype(font_path, font_size)
            except Exception as e:
                print(f"Error loading local font: {e}, using default font at size: {font_size}")
                return ImageFont.load_default()
        else:
            print(f"Font not found at {font_path}, attempting download...")
            new_path = self._ensure_font(font_path, font_type)
            if new_path:
                try:
                    print(f"Downloaded font: {new_path}, using style-based size: {font_size}")
                    return ImageFont.truetype(new_path, font_size)
                except Exception as e:
                    print(f"Error loading downloaded font: {e}, using default font")
                    return ImageFont.load_default()
            print(f"No font available, using default font at size: {font_size}")
            return ImageFont.load_default()

    def _draw_text(self, img, text, position, style="segment"):
        w, h = img.size
        draw = ImageDraw.Draw(img)
        
        if style == "hook":
            font_path = self.hook_font_path
            font_type = "hook"
            font_size = int(h * 0.05)
            color = "#FFFFFF"
            outline_color = "#000000"
            outline_width = 2
        elif style == "cta":
            font_path = self.cta_font_path
            font_type = "cta"
            font_size = int(h * 0.05)
            color = "#FFFFFF"  # unified color for CTA
            outline_color = "#000000"
            outline_width = 2
        else:
            font_path = self.segment_font_path
            font_type = "segment"
            font_size = int(h * 0.05)
            color = "#FFFFFF"
            outline_color = "#000000"
            outline_width = 1
        
        font = self._load_font(font_path, font_type, font_size)
        print(f"[_draw_text] style={style}, font_path={font_path}, font_size={font_size}")
        
        # Calculate text wrapping
        try:
            avg_char_width = draw.textlength("A", font=font)
        except AttributeError:
            # For older PIL versions
            try:
                avg_char_width = font.getlength("A")
            except AttributeError:
                # Even older PIL versions
                avg_char_width = font.getsize("A")[0]
            
        # Loosen the text wrapping to 0.9 of the width
        max_chars = max(10, int(w * 0.9 // avg_char_width))
        wrapped = textwrap.fill(text, width=max_chars)
        
        # Calculate text dimensions
        try:
            bbox = draw.textbbox((0, 0), wrapped, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            # For older PIL versions
            tw, th = draw.textsize(wrapped, font=font)
        
        # Center text horizontally and vertically
        x = (w - tw) // 2
        y = (h - th) // 2

        # Draw text with outline/shadow
        for offset_x, offset_y in [(-outline_width, -outline_width), 
                                  (outline_width, -outline_width), 
                                  (-outline_width, outline_width), 
                                  (outline_width, outline_width)]:
            draw.text((x + offset_x, y + offset_y), wrapped, font=font, 
                     fill=ImageColor.getrgb(outline_color) + (255,))

        # Draw the main text
        draw.text((x, y), wrapped, font=font, fill=ImageColor.getrgb(color) + (255,))
        
        return img
        
    def _get_video_frame(self, video_path, frame_time=0):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_duration = total_frames / fps if fps else 0
            frame_time = frame_time % total_duration
            frame_number = int(fps * frame_time)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Could not read frame at time {frame_time}s")
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            pil_image = Image.fromarray(frame_rgb)
            
            cap.release()
            
            return pil_image
            
        except Exception as e:
            print(f"Error extracting frame: {e}")
            dims = self.video_config["output_size"]
            return Image.new("RGB", (dims[0], dims[1]), (0, 0, 0))

    def generate_preview(self, csv_file, reel_id=None, segment_time=None, preview_type="all"):
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
            
        print(f"Generating preview for reel: {reel_id}")
            
        dims = self.video_config["output_size"]
        
        video_folder = self.video_config["video_folder"]
        if not os.path.exists(video_folder):
            raise FileNotFoundError(f"Video folder not found: {video_folder}")

        video = os.path.join(video_folder, "bg1.mp4")
        if not os.path.exists(video):
            for ext in (".mov", ".avi"): 
                p = video.replace(".mp4", ext)
                if os.path.exists(p): 
                    video = p
                    break
            else:
                templates = [f for f in os.listdir(video_folder) if f.endswith((".mp4", ".mov", ".avi"))]
                if not templates: 
                    raise FileNotFoundError("No video found in folder.")
                else:
                    video = os.path.join(video_folder, templates[0])
                    def _get_video_frame_func(tpos):
                        return self._get_video_frame(video, tpos)
                    get_frame_func = _get_video_frame_func
        else:
            def _get_video_frame_func(tpos):
                return self._get_video_frame(video, tpos)
            get_frame_func = _get_video_frame_func
                
        print(f"Using background video: {video}")
        
        tim = self.video_config.get("segment_timings", {})
        
        segments_to_preview = []
        
        if preview_type == "all":
            if pd.notna(row.get("hook")):
                segments_to_preview.append(("hook", row["hook"], tim.get("hook", {}).get("start", 0)))
            
            i = 1
            while f"segment{i}" in row:
                txt = row.get(f"segment{i}")
                if pd.notna(txt) and txt:
                    st = self._parse_time(row.get(f"segment{i}_start"), tim.get(f"segment{i}", {}).get("start", 0))
                    segments_to_preview.append((f"segment{i}", txt, st))
                i += 1
            
            if pd.notna(row.get("cta")):
                cta_st = self._parse_time(row.get("cta_start"), tim.get("cta", {}).get("start", 0))
                segments_to_preview.append(("cta", row["cta"], cta_st))
        else:
            if preview_type == "hook" and pd.notna(row.get("hook")):
                segments_to_preview.append(("hook", row["hook"], tim.get("hook", {}).get("start", 0)))
            elif preview_type == "cta" and pd.notna(row.get("cta")):
                cta_st = self._parse_time(row.get("cta_start"), tim.get("cta", {}).get("start", 0))
                segments_to_preview.append(("cta", row["call_to_action"], cta_st))
            elif preview_type.startswith("segment") and pd.notna(row.get(preview_type)):
                st = self._parse_time(row.get(f"{preview_type}_start"), 
                                     tim.get(preview_type, {}).get("start", 0))
                segments_to_preview.append((preview_type, row[preview_type], st))
        
        if not segments_to_preview:
            raise ValueError(f"No segments found to preview for type '{preview_type}'")
        
        preview_paths = []
        
        if segment_time is not None:
            segment_time = segment_time % 60

        for segment_id, text, time_pos in segments_to_preview:
            time_pos = time_pos % 60
            if segment_time is not None and abs(time_pos - segment_time) > 0.5:
                continue
                
            frame = get_frame_func(time_pos)  # use the segment's time for all
            
            tw, th = dims
            frame = frame.resize((tw, th), Image.LANCZOS)
            
            text_position = {"y": th // 2}
            style = "hook" if segment_id == "hook" else "cta" if segment_id == "cta" else "segment"
            frame_with_text = self._draw_text(frame, text, text_position, style=style)
            
            output_path = os.path.join(self.preview_dir, f"{reel_id}_{segment_id}.png")
            frame_with_text.save(output_path)
            print(f"Saved preview to {output_path}")
            preview_paths.append(output_path)
            
        return preview_paths
    
    def preview_all_text_at_once(self, csv_file, reel_id=None, background_time=0):
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
            
        print(f"Generating composite preview for reel: {reel_id}")
            
        dims = self.video_config["output_size"]
        tw, th = dims
        
        video_folder = self.video_config["video_folder"]
        if not os.path.exists(video_folder):
            raise FileNotFoundError(f"Video folder not found: {video_folder}")

        video = os.path.join(video_folder, "bg1.mp4")
        if not os.path.exists(video):
            for ext in (".mov", ".avi"): 
                p = video.replace(".mp4", ext)
                if os.path.exists(p): 
                    video = p
                    break
            else:
                templates = [f for f in os.listdir(video_folder) if f.endswith((".mp4", ".mov", ".avi"))]
                if not templates: 
                    raise FileNotFoundError("No video found in folder.")
                else:
                    video = os.path.join(video_folder, templates[0])
                    def _get_video_frame_func(tpos):
                        return self._get_video_frame(video, tpos)
                    get_frame_func = _get_video_frame_func
        else:
            def _get_video_frame_func(tpos):
                return self._get_video_frame(video, tpos)
            get_frame_func = _get_video_frame_func

        print(f"Using background video: {video}")
        background_time = background_time % 60
        frame = get_frame_func(background_time)
        frame = frame.resize((tw, th), Image.LANCZOS)
        
        if pd.notna(row.get("hook")):
            hook_position = {"y": int(th*0.2)}
            frame = self._draw_text(frame, row["hook"], hook_position, style="hook")
        
        segment_texts = []
        i = 1
        while f"segment{i}" in row:
            txt = row.get(f"segment{i}")
            if pd.notna(txt) and txt:
                segment_texts.append((f"segment{i}", txt))
            i += 1
        
        for idx, (segment_id, text) in enumerate(segment_texts):
            segment_position = {"y": int(th*0.4 + idx*(th*0.1))}
            frame = self._draw_text(frame, text, segment_position, style="segment")
        
        if pd.notna(row.get("cta")):
            cta_position = {"y": int(th*0.8)}
            frame = self._draw_text(frame, row["cta"], cta_position, style="cta")
        
        output_path = os.path.join(self.preview_dir, f"{reel_id}_composite.png")
        frame.save(output_path)
        print(f"Saved composite preview to {output_path}")
        
        return output_path