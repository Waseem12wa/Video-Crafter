"""
Subtitle Design 7 - Bold White on Blue Box
Same as Design 2 but with blue background instead of black.
Bold white text with solid blue rounded background box.
"""

from video_overlay_script import SubtitleDesign
import cv2


def get_design(aspect_ratio: str) -> SubtitleDesign:
    """
    Get Subtitle Design 7 configuration.
    Bold white text on solid blue rounded rectangle background.
    
    Args:
        aspect_ratio: Video aspect ratio (e.g., '4:5', '9:16')
    
    Returns:
        SubtitleDesign object with Design 7 configuration
    """
    if aspect_ratio == "9:16":
        # Vertical video design
        return SubtitleDesign(
            bar_color=(215, 150, 60),      # Blue background (BGR: #3c96d7 = RGB 60,150,215)
            bar_opacity=0.95,              # Very dark (95% opacity - less transparent)
            text_color=(255, 255, 255),    # White text
            text_scale=1.5,                # Larger scale for bold appearance
            text_thickness=4,              # Very thick/bold text
            outline_color=(0, 0, 0),       # No outline needed
            outline_thickness=0,           # No outline
            highlight_color=(255, 255, 255), # White highlight (same as text)
            highlight_text_color=(255, 255, 255), # White text
            margin=0,
            margin_x=20,                   # Good horizontal padding
            margin_y=15,                   # Good vertical padding
            bottom_margin=400,             # Position higher for vertical
            max_line_width_ratio=0.85,     # Allow wider text
            line_spacing=8,                # Tight line spacing
            corner_radius=15,              # Rounded corners
            highlight_padding=(0, 0),      # No extra highlight padding
            box_shadow_offset=(0, 0),      # No shadow
            box_shadow_blur=0,
            box_shadow_alpha=0.0,
            shadow_color=(0, 0, 0),
            shadow_offset=(0, 0),          # No text shadow
            shadow_thickness=0,
            font=cv2.FONT_HERSHEY_SIMPLEX,
            font_path="fonts/Montserrat-ExtraBold.ttf", # Same font as Design 2
            font_size_px=60,               # Large font size
        )
    else:
        # 4:5 and other aspect ratios
        return SubtitleDesign(
            bar_color=(215, 150, 60),      # Blue background (BGR: #3c96d7 = RGB 60,150,215)
            bar_opacity=0.95,              # Very dark (95% opacity - less transparent)
            text_color=(255, 255, 255),    # White text
            text_scale=1.5,                # Larger scale for bold appearance
            text_thickness=4,              # Very thick/bold text
            outline_color=(0, 0, 0),       # No outline needed
            outline_thickness=0,           # No outline
            highlight_color=(255, 255, 255), # White highlight (same as text)
            highlight_text_color=(255, 255, 255), # White text
            margin=0,
            margin_x=20,                   # Good horizontal padding inside box
            margin_y=15,                   # Good vertical padding inside box
            bottom_margin=30,              # Position from bottom
            max_line_width_ratio=0.85,     # Allow wider text blocks
            line_spacing=8,                # Tight line spacing
            corner_radius=15,              # Rounded corners on blue box
            highlight_padding=(0, 0),      # No extra highlight padding
            box_shadow_offset=(0, 0),      # No shadow
            box_shadow_blur=0,
            box_shadow_alpha=0.0,
            shadow_color=(0, 0, 0),
            shadow_offset=(0, 0),          # No text shadow
            shadow_thickness=0,
            font=cv2.FONT_HERSHEY_SIMPLEX,
            font_path="fonts/Montserrat-ExtraBold.ttf", # Same font as Design 2
            font_size_px=60,               # Large font size for bold appearance
        )
