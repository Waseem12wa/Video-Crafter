"""
Subtitle Design 3 - Bold Black on Beige Bar
Bold black Impact font text on light beige/cream colored bar.
"""

from video_overlay_script import SubtitleDesign
import cv2


def get_design(aspect_ratio: str) -> SubtitleDesign:
    """
    Get Subtitle Design 3 configuration.
    Bold black Impact font on light beige bar background.
    
    Args:
        aspect_ratio: Video aspect ratio (e.g., '4:5', '9:16')
    
    Returns:
        SubtitleDesign object with Design 3 configuration
    """
    if aspect_ratio == "9:16":
        # Vertical video design
        return SubtitleDesign(
            bar_color=(255, 255, 255),     # White background (BGR)
            bar_opacity=1.0,               # Full opacity (solid)
            text_color=(0, 0, 0),          # Black text
            text_scale=1.5,                # Same as Design 2
            text_thickness=4,              # Same as Design 2
            outline_color=(0, 0, 0),       # No outline
            outline_thickness=0,
            highlight_color=(0, 0, 0),     # Black (same as text)
            highlight_text_color=(0, 0, 0), # Black text
            margin=0,
            margin_x=20,                   # Same padding as Design 2
            margin_y=15,                   # Same padding as Design 2
            bottom_margin=400,             # Position higher for vertical
            max_line_width_ratio=0.85,     # Same as Design 2
            line_spacing=8,                # Same as Design 2
            corner_radius=15,              # Same rounded corners as Design 2
            highlight_padding=(0, 0),      # No extra highlight padding
            box_shadow_offset=(0, 0),      # No shadow
            box_shadow_blur=0,
            box_shadow_alpha=0.0,
            shadow_color=(0, 0, 0),
            shadow_offset=(0, 0),          # No text shadow
            shadow_thickness=0,
            font=cv2.FONT_HERSHEY_SIMPLEX,
            font_path="fonts/Montserrat-ExtraBold.ttf",  # Montserrat ExtraBold font
            font_size_px=60,               # Font size
        )
    else:
        # 4:5 and other aspect ratios
        return SubtitleDesign(
            bar_color=(255, 255, 255),     # White background (BGR)
            bar_opacity=1.0,               # Full opacity (solid)
            text_color=(0, 0, 0),          # Black text
            text_scale=1.5,                # Same as Design 2
            text_thickness=4,              # Same as Design 2
            outline_color=(0, 0, 0),       # No outline
            outline_thickness=0,
            highlight_color=(0, 0, 0),     # Black (same as text)
            highlight_text_color=(0, 0, 0), # Black text
            margin=0,
            margin_x=20,                   # Same padding as Design 2
            margin_y=15,                   # Same padding as Design 2
            bottom_margin=30,              # Position from bottom
            max_line_width_ratio=0.85,     # Same as Design 2
            line_spacing=8,                # Same as Design 2
            corner_radius=15,              # Same rounded corners as Design 2
            highlight_padding=(0, 0),      # No extra highlight padding
            box_shadow_offset=(0, 0),      # No shadow
            box_shadow_blur=0,
            box_shadow_alpha=0.0,
            shadow_color=(0, 0, 0),
            shadow_offset=(0, 0),          # No text shadow
            shadow_thickness=0,
            font=cv2.FONT_HERSHEY_SIMPLEX,
            font_path="fonts/Montserrat-ExtraBold.ttf",  # Montserrat ExtraBold font
            font_size_px=60,               # Font size
        )
