
from typing import List
from video_overlay_script import transcribe_audio_whisperx

# =============================================================================
# Robust Subtitle/Audio Alignment System
# =============================================================================

def build_subtitle_sentences_with_whisperx(audio_path: str, lines: List[str], model_size: str = "base", language: str = "en") -> tuple[List[dict], List[dict]]:
    """
    Build subtitle sentences using WhisperX for robust, waveform-aligned timing.
    Args:
        audio_path: Path to the audio file.
        lines: List of subtitle lines (sentences).
        model_size: WhisperX model size (default: "base").
        language: Language code (default: "en").
    Returns:
        Tuple of (transcript, subtitles) with accurate timing.
    """
    # Get word-level transcript with precise timings
    transcript = transcribe_audio_whisperx(audio_path, model_size, language)
    subtitles = []
    word_idx = 0
    for line in lines:
        tokens = [tok for tok in line.split() if tok]
        if not tokens:
            continue
        start_word = word_idx
        end_word = word_idx + len(tokens) - 1
        # Find start/end time from transcript
        if end_word < len(transcript):
            start_time = transcript[start_word]["start_time"]
            end_time = transcript[end_word]["end_time"]
        else:
            # Fallback: use last available word
            start_time = transcript[start_word]["start_time"] if start_word < len(transcript) else 0.0
            end_time = transcript[-1]["end_time"] if transcript else 0.0
        subtitles.append({
            "text": line,
            "start_word": start_word,
            "end_word": end_word,
            "word_count": len(tokens),
            "start_time": start_time,
            "end_time": end_time,
        })
        word_idx += len(tokens)
    return transcript, subtitles
"""
Subtitle Design 1 - Default Design
This is the current default subtitle design.
Also contains subtitle processing functions.
"""

from video_overlay_script import SubtitleSentence, SubtitleDesign
from typing import List
import cv2
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


# =============================================================================
# Subtitle Processing Functions
# =============================================================================

def build_subtitle_sentences_from_lines(lines: List[str], default_word_duration: float = 0.5) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    This function is deprecated and will be replaced by a robust alignment system.
    """
    raise NotImplementedError("This function is deprecated. Use the new robust alignment system.")


def build_subtitle_sentences_from_dict(subtitle_sentences: List[Any], is_dummy_transcript: bool = False) -> List[SubtitleSentence]:
    """
    Build SubtitleSentence objects from dictionary data.
    
    Args:
        subtitle_sentences: List of subtitle sentence data (dicts or strings)
        is_dummy_transcript: Whether the transcript is a dummy transcript
    
    Returns:
        List of SubtitleSentence objects
    """
    def _safe_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    sentences = []
    if subtitle_sentences and not is_dummy_transcript:
        # Only use frontend sentences if we trust the transcript
        for s in subtitle_sentences:
            if isinstance(s, dict):
                sentences.append(SubtitleSentence(
                    text=s.get('text', ''),
                    phrase=s.get('text', ''),
                    occurrence=s.get('occurrence', 1),
                    start_word=_safe_int(s.get('start_word')),
                    end_word=_safe_int(s.get('end_word')),
                ))
            elif isinstance(s, str):
                sentences.append(SubtitleSentence(text=s, phrase=s))

    return sentences


def serialize_subtitle_sentences(subtitle_sentences: List[Any]) -> List[Dict[str, Any]]:
    """
    Convert subtitle sentences to serializable format for JSON storage.
    
    Args:
        subtitle_sentences: List of SubtitleSentence objects, dicts, or strings
    
    Returns:
        List of serializable subtitle sentence dictionaries
    """
    serializable_subtitles = []
    for s in subtitle_sentences:
        if isinstance(s, SubtitleSentence):
            serializable_subtitles.append({
                'text': s.text,
                'phrase': s.phrase if s.phrase else s.text,
                'occurrence': s.occurrence,
                'start_word': s.start_word if hasattr(s, 'start_word') else None,
                'end_word': s.end_word if hasattr(s, 'end_word') else None,
            })
        elif isinstance(s, dict):
            serializable_subtitles.append(s)
        else:
            # String or other type
            serializable_subtitles.append({
                'text': str(s),
                'phrase': str(s),
                'occurrence': 1
            })
    
    return serializable_subtitles


# =============================================================================
# Design Configuration
# =============================================================================

def get_design(aspect_ratio: str):
    """
    Get Subtitle Design 1 configuration.
    
    Args:
        aspect_ratio: Video aspect ratio (e.g., '4:5', '9:16')
    
    Returns:
        SubtitleDesign object with Design 1 configuration
    """
    if aspect_ratio == "9:16":
        # TikTok-style portrait/vertical video design
        return SubtitleDesign(
            bar_color=(0, 0, 0),
            bar_opacity=0.0,
            text_color=(255, 255, 255),
            text_scale=1.2,
            text_thickness=3,
            outline_color=(0, 0, 0),
            outline_thickness=5,
            highlight_color=(255, 230, 90),
            highlight_text_color=(255, 255, 255),
            margin=0,
            margin_x=6,
            margin_y=0,
            bottom_margin=400,
            max_line_width_ratio=0.72,
            line_spacing=4,
            corner_radius=0,
            highlight_padding=(4, 2),
            box_shadow_offset=(0, 0),
            box_shadow_blur=0,
            box_shadow_alpha=0.0,
            shadow_color=(0, 0, 0),
            shadow_offset=(0, 0),
            shadow_thickness=0,
            font=cv2.FONT_HERSHEY_SIMPLEX,
            font_path="fonts/Poppins-SemiBold.ttf",
            font_size_px=54,
        )
    else:
        # Default 4:5 design
        return SubtitleDesign(
            bar_color=(0, 0, 0),
            bar_opacity=0.75,
            text_color=(255, 255, 255),
            text_scale=1.25,
            text_thickness=2,
            outline_color=(0, 0, 0),
            outline_thickness=0,
            highlight_color=(255, 230, 90),
            highlight_text_color=(255, 255, 255),
            margin=0,
            margin_x=6,
            margin_y=0,
            bottom_margin=30,
            max_line_width_ratio=0.72,
            line_spacing=10,
            corner_radius=4,
            highlight_padding=(3, 1),
            box_shadow_offset=(8, 10),
            box_shadow_blur=25,
            box_shadow_alpha=0.55,
            shadow_color=(0, 0, 0),
            shadow_offset=(8, 10),
            shadow_thickness=10,
            font=cv2.FONT_HERSHEY_SIMPLEX,
            font_path="fonts/Montserrat-SemiBold.ttf",
            font_size_px=54,
        )
