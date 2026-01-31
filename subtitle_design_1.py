"""
Subtitle Design 1 - Default Design
This is the current default subtitle design.
Also contains subtitle processing functions.
"""

from video_overlay_script import SubtitleSentence, SubtitleDesign
import cv2
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


# =============================================================================
# Subtitle Processing Functions
# =============================================================================

def build_subtitle_sentences_from_lines(lines: List[str], default_word_duration: float = 0.5) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build subtitle sentences from text lines.
    
    Args:
        lines: List of text lines to convert to subtitles
        default_word_duration: Duration for each word in seconds (default: 0.5)
    
    Returns:
        Tuple of (transcript, subtitles) where:
        - transcript: List of word dictionaries with timing information
        - subtitles: List of subtitle sentence dictionaries
    """
    transcript = []
    subtitles = []
    word_cursor = 0

    for line in lines:
        tokens = [tok for tok in line.split() if tok]
        if not tokens:
            continue

        start_word = word_cursor
        for idx, tok in enumerate(tokens):
            t0 = (word_cursor + idx) * default_word_duration
            t1 = t0 + default_word_duration
            transcript.append({"word": tok, "start_time": t0, "end_time": t1})
        word_cursor += len(tokens)
        end_word = word_cursor - 1

        subtitles.append(
            {
                "text": line,
                "start_word": start_word,
                "end_word": end_word,
                "word_count": len(tokens),
            }
        )

    return transcript, subtitles


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
