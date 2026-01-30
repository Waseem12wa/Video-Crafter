"""
Subtitle Design 6 - Bold White Text with Transparent Box
Same as Design 2 but with transparent background instead of black.
Bold white text with rounded transparent background.
"""

from video_overlay_script import SubtitleDesign
import cv2


def get_design(aspect_ratio: str) -> SubtitleDesign:
    """
    Get Subtitle Design 6 configuration.
    Bold white text with transparent rounded rectangle background.
    
    Args:
        aspect_ratio: Video aspect ratio (e.g., '4:5', '9:16')
    
    Returns:
        SubtitleDesign object with Design 6 configuration
    """
    if aspect_ratio == "9:16":
        # Vertical video design
        return SubtitleDesign(
            bar_color=(0, 0, 0),           # Black background
            bar_opacity=0.75,              # Darker semi-transparent (75% opacity)
            text_color=(255, 255, 255),    # White text
            text_scale=1.5,                # Same as Design 2
            text_thickness=4,              # Same as Design 2
            outline_color=(0, 0, 0),       # No outline needed (box provides visibility)
            outline_thickness=0,           # No outline
            highlight_color=(255, 255, 255), # White highlight (same as text)
            highlight_text_color=(255, 255, 255), # White text
            margin=0,
            margin_x=20,                   # Same padding as Design 2
            margin_y=15,                   # Same padding as Design 2
            bottom_margin=400,             # Same position as Design 2
            max_line_width_ratio=0.85,     # Same as Design 2
            line_spacing=8,                # Same as Design 2
            corner_radius=8,               # Less rounded corners (8px instead of 15px)
            highlight_padding=(0, 0),      # No extra highlight padding
            box_shadow_offset=(0, 0),      # No shadow
            box_shadow_blur=0,
            box_shadow_alpha=0.0,
            shadow_color=(0, 0, 0),
            shadow_offset=(0, 0),          # No text shadow
            shadow_thickness=0,
            font=cv2.FONT_HERSHEY_SIMPLEX,
            font_path="fonts/Montserrat-ExtraBold.ttf", # Same font as Design 2
            font_size_px=60,               # Same font size as Design 2
        )
    else:
        # 4:5 and other aspect ratios
        return SubtitleDesign(
            bar_color=(0, 0, 0),           # Black background
            bar_opacity=0.75,              # Darker semi-transparent (75% opacity)
            text_color=(255, 255, 255),    # White text
            text_scale=1.5,                # Same as Design 2
            text_thickness=4,              # Same as Design 2
            outline_color=(0, 0, 0),       # No outline needed (box provides visibility)
            outline_thickness=0,           # No outline
            highlight_color=(255, 255, 255), # White highlight (same as text)
            highlight_text_color=(255, 255, 255), # White text
            margin=0,
            margin_x=20,                   # Same padding as Design 2
            margin_y=15,                   # Same padding as Design 2
            bottom_margin=30,              # Same position as Design 2
            max_line_width_ratio=0.85,     # Same as Design 2
            line_spacing=8,                # Same as Design 2
            corner_radius=8,               # Less rounded corners (8px instead of 15px)
            highlight_padding=(0, 0),      # No extra highlight padding
            box_shadow_offset=(0, 0),      # No shadow
            box_shadow_blur=0,
            box_shadow_alpha=0.0,
            shadow_color=(0, 0, 0),
            shadow_offset=(0, 0),          # No text shadow
            shadow_thickness=0,
            font=cv2.FONT_HERSHEY_SIMPLEX,
            font_path="fonts/Montserrat-ExtraBold.ttf", # Same font as Design 2
            font_size_px=60,               # Same font size as Design 2
        )
