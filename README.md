
# 🎞️ Reelery – AI-Powered Bulk Reels Generator

**Reelery** is a powerful automated Python pipeline that transforms CSV data into engaging short-form video reels with AI-generated voiceovers, animated subtitles, and randomized background videos. Generate professional-quality reels in under 4 minutes each.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/reelery.git
cd reelery

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Add your content and background videos
# Edit inputs/scripts/content.csv with your content
# Add .mp4 files to inputs/videos/

# Generate all reels
python main.py --all

```

## ✨ Features

-   🎙️ **AI Voiceover Generation** - Natural-sounding voice synthesis using Microsoft's edge-tts
-   📝 **Animated Subtitles** - Syllable-aware timing with customizable styling
-   🎥 **Random Background Videos** - Automatically selects from your video library
-   📊 **CSV-Based Workflow** - Simple spreadsheet-based content management
-   🧹 **Automatic Cleanup** - Handles temporary files and resource management
-   🖼️ **Preview Generation** - Create text-only previews for content review
-   ⚙️ **Highly Configurable** - Customize fonts, colors, timing, and layouts
-   🔄 **Batch Processing** - Generate hundreds of reels with a single command

## 🛠️ Installation

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Install FFmpeg (required for video processing)
# Windows: choco install ffmpeg
# macOS: brew install ffmpeg
# Linux: sudo apt install ffmpeg

```

## 📝 Content Setup

### 1. Prepare Your Content CSV

Create or edit `inputs/scripts/content.csv`:

```csv
reelid,hook
reel_001,Here's one tip they don't tell you about money...
reel_002,This is what successful people do before 8 AM...
reel_003,The biggest mistake people make when starting a business...

```

**CSV Fields:**

-   `reelid`: Unique identifier for each reel (used as filename)
-   `hook`: The main content/script for the reel

### 2. Add Background Videos

Place your `.mp4` background videos in `inputs/videos/`:

-   Recommended: 1080x1920 (9:16 aspect ratio)
-   Duration: 30 seconds or longer
-   Format: MP4, H.264 codec
-   One video will be randomly selected per reel

## 🔧 API Reference

### Command Line Arguments

Argument

Description

Example

`--all`

Generate all reels from CSV

`python main.py --all`

`--reelid <id>`

Generate specific reel

`python main.py --reelid reel_001`

`--skip-tts`

Skip TTS generation

`python main.py --all --skip-tts`

`--only-tts`

Generate audio only

`python main.py --all --only-tts`

`--preview`

Generate text previews

`python main.py --all --preview`

`--preview-type <type>`

Specify preview type

`python main.py --reelid reel_001 --preview --preview-type hook`

`--composite`

Generate composite preview

`python main.py --all --preview --composite`

`--download-font`

Download missing fonts

`python main.py --download-font`

### Preview Types

-   `hook`: Main content preview
-   `segment1`, `segment2`, etc.: Specific content segments
-   `all`: All segments combined

### Examples

```bash
# Generate all reels from CSV
python main.py --all

# Generate a specific reel
python main.py --reelid reel_001

# Generate only audio files (no video)
python main.py --all --only-tts

# Generate preview images
python main.py --all --preview

# Skip TTS if audio files already exist
python main.py --all --skip-tts

# Generate composite preview with all text
python main.py --reelid reel_001 --preview --composite

# Generate specific preview type
python main.py --reelid reel_001 --preview --preview-type hook

```

## ⚙️ Configuration

Edit `config.json` to customize your reels:

```json
{
  "video": {
    "width": 1080,
    "height": 1920,
    "fps": 30,
    "duration_padding": 1.0
  },
  "audio": {
    "voice": "en-US-AriaNeural",
    "rate": "+0%",
    "volume": "+0%"
  },
  "subtitles": {
    "font_size": 60,
    "font_color": "#FFFFFF",
    "stroke_color": "#000000",
    "stroke_width": 2,
    "position": "center"
  }
}

```

### Available Voice Options

-   `en-US-AriaNeural` (Female, conversational)
-   `en-US-DavisNeural` (Male, professional)
-   `en-US-JennyNeural` (Female, friendly)
-   `en-US-GuyNeural` (Male, casual)
-   And many more! See [edge-tts documentation](https://github.com/rany2/edge-tts#voice-list)

### Subtitle Styling

-   Colors: Use hex codes (#FFFFFF) or color names
-   Positioning: top, center, bottom, or custom coordinates
-   Animation: Fade in/out, slide, or custom effects

## 🧠 Tech Stack

-   **Python 3.8+** - Core runtime
-   **edge-tts** - AI voice synthesis
-   **moviepy** - Video creation and editing
-   **torchaudio + librosa** - Audio processing and silence trimming
-   **pandas** - CSV parsing and data handling
-   **matplotlib + Pillow** - Image and text rendering
-   **FFmpeg** - Video processing backend

----------

**Made with ❤️ by the Reelery Team**

⭐ **Star this repo** if you find it useful!
