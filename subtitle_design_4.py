"""
Subtitle Design 5 - Gradient Highlight Style
Bold black text on vibrant yellow/gold background with rounded corners and shadow.
Modern, eye-catching style perfect for video ads.
"""

from video_overlay_script import SubtitleDesign
import cv2


def get_design(aspect_ratio: str) -> SubtitleDesign:
    """
    Get Subtitle Design 5 configuration.
    Bold black text on yellow/gold gradient background - attractive and unique.
    
    Args:
        aspect_ratio: Video aspect ratio (e.g., '4:5', '9:16')
    
    Returns:
        SubtitleDesign object with Design 5 configuration
    """
    if aspect_ratio == "9:16":
        # Vertical video design
        return SubtitleDesign(
            bar_color=(0, 200, 255),       # Bright yellow/gold background (BGR: gold color)
            bar_opacity=1.0,               # Full opacity (solid)
            text_color=(0, 0, 0),          # Black text for high contrast
            text_scale=1.6,                # Large scale
            text_thickness=5,              # Very thick/bold text
            outline_color=(0, 0, 0),       # No outline needed (high contrast already)
            outline_thickness=0,
            highlight_color=(255, 230, 90), # Lighter yellow for highlights
            highlight_text_color=(0, 0, 0), # Black text
            margin=0,
            margin_x=18,                   # Good horizontal padding
            margin_y=10,                   # Good vertical padding
            bottom_margin=400,             # Position higher for vertical
            max_line_width_ratio=0.85,     # Allow wider text
            line_spacing=8,                # Comfortable line spacing
            corner_radius=12,              # Rounded corners for modern look
            highlight_padding=(0, 0),      # No extra highlight padding
            box_shadow_offset=(6, 8),      # Drop shadow for depth
            box_shadow_blur=20,            # Soft shadow blur
            box_shadow_alpha=0.45,         # Medium shadow opacity
            shadow_color=(0, 0, 0),
            shadow_offset=(0, 0),          # No text shadow (background provides contrast)
            shadow_thickness=0,
            font=cv2.FONT_HERSHEY_SIMPLEX,
            font_path="fonts/Montserrat-ExtraBold.ttf",  # Bold, modern font
            font_size_px=62,               # Large font size
        )
    else:
        # 4:5 and other aspect ratios
        return SubtitleDesign(
            bar_color=(0, 200, 255),       # Bright yellow/gold background (BGR: gold color)
            bar_opacity=1.0,               # Full opacity (solid)
            text_color=(0, 0, 0),          # Black text for high contrast
            text_scale=1.6,                # Large scale
            text_thickness=5,              # Very thick/bold text
            outline_color=(0, 0, 0),       # No outline needed (high contrast already)
            outline_thickness=0,
            highlight_color=(255, 230, 90), # Lighter yellow for highlights
            highlight_text_color=(0, 0, 0), # Black text
            margin=0,
            margin_x=18,                   # Good horizontal padding
            margin_y=10,                   # Good vertical padding
            bottom_margin=30,              # Position from bottom
            max_line_width_ratio=0.85,     # Allow wider text blocks
            line_spacing=8,                # Comfortable line spacing
            corner_radius=12,              # Rounded corners for modern look
            highlight_padding=(0, 0),      # No extra highlight padding
            box_shadow_offset=(6, 8),      # Drop shadow for depth
            box_shadow_blur=20,            # Soft shadow blur
            box_shadow_alpha=0.45,         # Medium shadow opacity
            shadow_color=(0, 0, 0),
            shadow_offset=(0, 0),          # No text shadow (background provides contrast)
            shadow_thickness=0,
            font=cv2.FONT_HERSHEY_SIMPLEX,
            font_path="fonts/Montserrat-ExtraBold.ttf",  # Bold, modern font
            font_size_px=62,               # Large font size for bold appearance
        )
