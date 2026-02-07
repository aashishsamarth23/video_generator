# Video Generator

Generate a short (<30s) video from a prompt using an image library, open-source TTS narration, and MoviePy.

## How it works
1. Builds (or reuses) a local image library in `assets/images`.
2. Uses TF-IDF retrieval to select the most relevant images for the prompt.
3. Generates narration with Coqui TTS.
4. Adds simple transitions + a Ken Burns zoom to each image.
5. Outputs an `.mp4` video in the working directory.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python video_generator.py --prompt "A calm morning routine that builds into a burst of creativity"
```

### Optional parameters
```bash
python video_generator.py \
  --prompt "A hero's journey across a vivid landscape" \
  --model tts_models/en/ljspeech/tacotron2-DDC \
  --output my_video.mp4
```

## Image library
The script will auto-generate placeholder images and descriptions if `assets/images/metadata.json` is missing. Replace them with your own images and update `metadata.json` for custom content.
