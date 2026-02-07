#!/usr/bin/env python3
"""Generate a short video from a text prompt using images + TTS audio."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from moviepy.editor import AudioFileClip, CompositeVideoClip, ImageClip, concatenate_videoclips
from PIL import Image, ImageDraw, ImageFont
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from TTS.api import TTS
except ImportError:  # pragma: no cover - optional dependency at runtime
    TTS = None

ROOT = Path(__file__).resolve().parent
ASSETS_DIR = ROOT / "assets"
IMAGES_DIR = ASSETS_DIR / "images"
METADATA_PATH = IMAGES_DIR / "metadata.json"
DEFAULT_OUTPUT = ROOT / "generated_video.mp4"


@dataclass
class ImageEntry:
    path: Path
    description: str


class ImageLibrary:
    def __init__(self, entries: List[ImageEntry]):
        self.entries = entries
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([e.description for e in entries])

    def search(self, query: str, k: int = 5) -> List[ImageEntry]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        ranked_indices = np.argsort(scores)[::-1][:k]
        return [self.entries[i] for i in ranked_indices]


def ensure_image_library() -> ImageLibrary:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    if METADATA_PATH.exists():
        entries = load_metadata(METADATA_PATH)
        return ImageLibrary(entries)

    sample_data = [
        ("sunrise_city.png", "Sunrise over a modern city skyline with warm orange light."),
        ("forest_path.png", "A peaceful forest path with soft morning mist and tall trees."),
        ("ocean_waves.png", "Gentle ocean waves rolling onto a sandy shore at golden hour."),
        ("workspace.png", "A cozy creative workspace with a laptop, notebook, and coffee."),
        ("mountain_peak.png", "A dramatic mountain peak under a clear blue sky."),
        ("night_stars.png", "A calm night sky filled with stars above a quiet landscape."),
    ]

    entries: List[ImageEntry] = []
    for filename, description in sample_data:
        path = IMAGES_DIR / filename
        create_placeholder_image(path, description)
        entries.append(ImageEntry(path=path, description=description))

    save_metadata(entries, METADATA_PATH)
    return ImageLibrary(entries)


def create_placeholder_image(path: Path, caption: str) -> None:
    width, height = 1280, 720
    palette = [
        (255, 183, 77),
        (129, 199, 132),
        (79, 195, 247),
        (255, 241, 118),
        (158, 158, 158),
        (179, 157, 219),
    ]
    color = palette[hash(path.name) % len(palette)]
    image = Image.new("RGB", (width, height), color)
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 36)
    except OSError:
        font = ImageFont.load_default()

    wrapped = wrap_text(caption, max_chars=40)
    text_width, text_height = draw.multiline_textsize(wrapped, font=font, spacing=6)
    x = (width - text_width) / 2
    y = (height - text_height) / 2
    draw.rectangle(
        [x - 24, y - 24, x + text_width + 24, y + text_height + 24],
        fill=(0, 0, 0, 140),
    )
    draw.multiline_text((x, y), wrapped, font=font, fill=(255, 255, 255), spacing=6, align="center")
    image.save(path)


def wrap_text(text: str, max_chars: int) -> str:
    words = text.split()
    lines = []
    current = []
    for word in words:
        if sum(len(w) for w in current) + len(current) + len(word) > max_chars:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def save_metadata(entries: List[ImageEntry], path: Path) -> None:
    payload = [
        {"path": str(entry.path.relative_to(ROOT)), "description": entry.description}
        for entry in entries
    ]
    path.write_text(json.dumps(payload, indent=2))


def load_metadata(path: Path) -> List[ImageEntry]:
    payload = json.loads(path.read_text())
    entries = []
    for item in payload:
        entries.append(
            ImageEntry(path=ROOT / item["path"], description=item["description"])
        )
    return entries


def build_storyboard(prompt: str) -> str:
    prompt = prompt.strip()
    base = (
        f"{prompt} We set the scene with a calm opening, then build momentum, "
        "and finish with an uplifting takeaway." if prompt else "A gentle story unfolds with a calm opening, a rising middle, and a hopeful ending."
    )
    sentences = base.split(".")
    trimmed = ".".join(s.strip() for s in sentences if s.strip())
    words = trimmed.split()
    return " ".join(words[:70]).strip() + "."


def generate_voiceover(text: str, output_path: Path, model_name: str) -> None:
    if TTS is None:
        raise RuntimeError(
            "TTS is not installed. Please install the TTS package to generate audio."
        )
    tts = TTS(model_name=model_name, progress_bar=False, gpu=False)
    tts.tts_to_file(text=text, file_path=str(output_path))


def ken_burns(clip: ImageClip, zoom: float = 1.08) -> ImageClip:
    return clip.resize(lambda t: 1 + (zoom - 1) * (t / clip.duration))


def build_video(image_paths: List[Path], audio_path: Path, output_path: Path) -> None:
    audio = AudioFileClip(str(audio_path))
    duration = min(28.0, audio.duration)
    per_clip = max(2.0, duration / len(image_paths))

    clips = []
    for idx, image_path in enumerate(image_paths):
        clip = ImageClip(str(image_path)).set_duration(per_clip).resize(height=720)
        clip = ken_burns(clip, zoom=1.05 + (idx * 0.01))
        clip = clip.set_position("center")
        clip = clip.crossfadein(0.5).crossfadeout(0.5)
        clips.append(clip)

    video = concatenate_videoclips(clips, method="compose").set_audio(audio)
    final = CompositeVideoClip([video], size=video.size).set_duration(duration)
    final.write_videofile(str(output_path), fps=24, codec="libx264", audio_codec="aac")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a short video from a prompt.")
    parser.add_argument("--prompt", required=True, help="Prompt describing the desired video.")
    parser.add_argument(
        "--model",
        default="tts_models/en/ljspeech/tacotron2-DDC",
        help="Coqui TTS model name to use for narration.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output path for the generated video.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library = ensure_image_library()

    script = build_storyboard(args.prompt)
    selected = library.search(script, k=4)
    if not selected:
        raise RuntimeError("No images were found in the library.")

    audio_path = ROOT / "narration.wav"
    generate_voiceover(script, audio_path, args.model)

    build_video([entry.path for entry in selected], audio_path, Path(args.output))
    print(f"Video saved to {args.output}")


if __name__ == "__main__":
    main()
