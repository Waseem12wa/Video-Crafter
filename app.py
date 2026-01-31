"""
Flask web application for video editing frontend.
Allows users to upload videos, select transcript highlights, and process videos.
"""

import os
import json
import logging
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
import subprocess
import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

from flask import request

def log_request_context(tag: str):
    logger.info(
        "[%s] %s %s content_length=%s, remote_addr=%s, files=%s",
        tag,
        request.method,
        request.path,
        request.content_length,
        request.remote_addr,
        list(request.files.keys()),
    )

print("Importing video_overlay_script...")
from video_overlay_script import (
    ProjectConfig,
    HighlightAssignment,
    build_transcript,
    render_project,
    SubtitleSentence,
    get_subtitle_design_for_aspect_ratio
)
print("video_overlay_script imported.")

from typing import List

# Import subtitle processing functions from subtitle_design_1
from subtitle_design_1 import (
    build_subtitle_sentences_from_lines,
    build_subtitle_sentences_from_dict,
    serialize_subtitle_sentences
)

# Import subtitle design modules
import subtitle_design_1
import subtitle_design_2
import subtitle_design_3
import subtitle_design_4
import subtitle_design_5
import subtitle_design_6


def get_subtitle_design_by_selection(design_number: int, aspect_ratio: str):
    """
    Get subtitle design based on user selection.
    
    Args:
        design_number: Design number (1-5)
        aspect_ratio: Video aspect ratio (e.g., '4:5', '9:16')
    
    Returns:
        SubtitleDesign object for the selected design
    """
    design_map = {
        1: subtitle_design_1.get_design,
        2: subtitle_design_2.get_design,
        3: subtitle_design_3.get_design,
        4: subtitle_design_4.get_design,
        5: subtitle_design_5.get_design,
        6: subtitle_design_6.get_design,
    }
    
    # Default to design 1 if invalid selection
    design_func = design_map.get(design_number, subtitle_design_1.get_design)
    return design_func(aspect_ratio)
# Check if React build exists
USE_REACT_BUILD = os.path.exists('frontend/dist/index.html')

if USE_REACT_BUILD:
    # Serve React build
    app = Flask(__name__, static_folder='frontend/dist', static_url_path='')
else:
    # Serve traditional templates
    app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

# S3 Configuration
S3_BUCKET_NAME = 's3videocrafter'
S3_REGION = 'us-east-1'
AWS_ACCESS_KEY_ID = 'AKIA3ETPXFJGJVSFMZPX'
AWS_SECRET_ACCESS_KEY = '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'


# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs('clips', exist_ok=True)
os.makedirs('audio_files', exist_ok=True)

ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm'}
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'aac', 'm4a', 'flac', 'ogg'}

def allowed_file(filename, extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=S3_REGION
)


# S3 Upload Functions
def upload_file_to_s3(file_path, s3_key):
    """Upload a file to S3 bucket."""
    try:
        s3_client.upload_file(
            file_path,
            S3_BUCKET_NAME,
            s3_key,
            ExtraArgs={'ContentType': get_content_type(file_path)}
        )
        s3_url = f"https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/{s3_key}"
        return s3_url
    except ClientError as e:
        print(f"Error uploading {file_path} to S3: {e}")
        raise


def convert_audio_to_video(audio_path, output_video_path):
    """Convert audio to video with a black background using ffmpeg."""
    try:
        # Use a black background
        cmd = [
            'ffmpeg',
            '-y',
            '-f', 'lavfi',
            '-i', 'color=c=black:s=1280x720:r=25',
            '-i', audio_path,
            '-c:v', 'libx264',
            '-tune', 'stillimage',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-pix_fmt', 'yuv420p',
            '-shortest',
            output_video_path
        ]
        
        logger.info(f"Running conversion command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error converting audio to video: {e}")
        return False

def get_content_type(file_path):
    """Get content type based on file extension."""
    ext = Path(file_path).suffix.lower()
    content_types = {
        '.mp4': 'video/mp4',
        '.json': 'application/json',
        '.txt': 'text/plain',
        '.mp3': 'audio/mpeg',
        '.wav': 'audio/wav',
    }
    return content_types.get(ext, 'application/octet-stream')


def create_project_json(video_path, highlights, transcript, subtitle_sentences, 
                        aspect_ratio, output_filename=None, output_path=None):
    """Create a project JSON file with all project data."""
    # Convert subtitle sentences to serializable format using subtitle module
    serializable_subtitles = serialize_subtitle_sentences(subtitle_sentences)
    
    project_data = {
        'project_info': {
            'created_at': datetime.now().isoformat(),
            'video_path': video_path,
            'output_filename': output_filename,
            'output_path': output_path,
            'aspect_ratio': aspect_ratio,
        },
        'highlights': highlights,
        'transcript': transcript,
        'subtitle_sentences': serializable_subtitles,
        'statistics': {
            'total_highlights': len(highlights),
            'total_transcript_words': len(transcript),
            'total_subtitle_sentences': len(subtitle_sentences),
        }
    }
    return project_data


# def render_project_with_transcript(config: ProjectConfig, transcript: list):
#     """
#     Render project using an existing transcript instead of regenerating it.
#     This avoids calling Whisper again which is slow and unnecessary.
#     """
#     from video_overlay_script import (
#         map_assignments_to_segments,
#         process_video_with_overlays,
#         merge_audio_tracks,
#         generate_default_subtitle_segments,
#         HAVE_MOVIEPY
#     )

#     # Use the provided transcript instead of calling build_transcript
#     highlight_segments = map_assignments_to_segments(
#         transcript, config.highlight_assignments
#     )

#     any_segment_music = any(
#         assignment.music_path for assignment in config.highlight_assignments
#     )
#     needs_audio_merge = HAVE_MOVIEPY and (
#         config.preserve_audio or bool(config.global_music_path) or any_segment_music
#     )
#     final_output_path = config.output_path
#     silent_output_path = final_output_path

#     # Generate subtitle segments
#     subtitle_segments = config.subtitle_segments
#     if subtitle_segments is None:
#         subtitle_segments = generate_default_subtitle_segments(
#             transcript, highlight_segments
#         )

#     if needs_audio_merge:
#         root, ext = os.path.splitext(final_output_path)
#         ext = ext or ".mp4"
#         silent_output_path = f"{root}.silent{ext}"

#     # Render video with overlays
#     process_video_with_overlays(
#         config.main_video_path,
#         transcript,
#         highlight_segments,
#         config.subtitle_design,
#         silent_output_path,
#         subtitle_segments=subtitle_segments,
#         custom_subtitles=None,
#     )

#     # Merge audio if needed
#     if needs_audio_merge:
#         merge_audio_tracks(
#             silent_output_path,
#             config.main_video_path,
#             transcript,
#             highlight_segments,
#             final_output_path,
#             preserve_main_audio=config.preserve_audio,
#             global_music_path=config.global_music_path,
#             global_music_volume=config.global_music_volume,
#         )

#         if os.path.exists(silent_output_path) and silent_output_path != final_output_path:
#             os.remove(silent_output_path)

#     return {
#         "output_path": final_output_path,
#         "transcript": transcript,
#         "highlight_segments": highlight_segments,
#     }

ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'aac', 'm4a', 'flac', 'ogg', 'm4a'}


def convert_audio_to_video(audio_path: str, output_video_path: str) -> bool:
    """
    Convert an audio file to a video file (MP4) with a black background.
    Target resolution: 1080x1920 (9:16 vertical)
    """
    try:
        command = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=1080x1920:r=30",
            "-i", audio_path,
            "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            output_video_path
        ]
        logger.info(f"[CONVERT] Running: {' '.join(command)}")
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"[CONVERT] FFmpeg failed: {e.stderr.decode()}")
        return False
    except Exception as e:
        logger.error(f"[CONVERT] Error: {e}")
        return False


def allowed_file(filename, allowed_extensions):
    """Check if file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


@app.route('/')
def index():
    """Render the main page."""
    if USE_REACT_BUILD:
        return send_from_directory(app.static_folder, 'index.html')
    return render_template('index.html')


@app.route('/test-route')
def test_route():
    """Test route to verify server is responding."""
    return jsonify({'message': 'Server is working!', 'routes': ['upload-video', 'upload-video-with-txt']})


@app.route('/upload-video', methods=['POST'])
def upload_video():
    """Handle main video upload and generate transcript."""
    log_request_context("UPLOAD_VIDEO")

    if 'video' not in request.files:
        logger.warning("[UPLOAD_VIDEO] No 'video' file in request")
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    is_video = allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS)
    is_audio = allowed_file(file.filename, ALLOWED_AUDIO_EXTENSIONS)

    if not (is_video or is_audio):
        return jsonify({'error': 'Invalid file type. Please upload a video or audio file.'}), 400

    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.info("[UPLOAD_VIDEO] Saving file as %s", save_path)
        file.save(save_path)
        
        # Determine final video path to process
        video_path = save_path
        
        if is_audio:
            # Convert audio to video
            filename_no_ext = os.path.splitext(filename)[0]
            converted_video_filename = f"{filename_no_ext}_converted.mp4"
            converted_video_path = os.path.join(app.config['UPLOAD_FOLDER'], converted_video_filename)
            
            logger.info(f"[UPLOAD_VIDEO] Converting audio {filename} to video {converted_video_filename}...")
            if convert_audio_to_video(save_path, converted_video_path):
                video_path = converted_video_path
                logger.info("[UPLOAD_VIDEO] Conversion successful")
            else:
                return jsonify({'error': 'Failed to convert audio file to video'}), 500

        logger.info(
            "[UPLOAD_VIDEO] Using video source (%s), size ~%.2f MB",
            os.path.basename(video_path),
            os.path.getsize(video_path) / (1024 * 1024),
        )

        # Generate transcript using Whisper
        whisper_model = request.form.get('whisper_model', 'base')
        transcript = build_transcript(video_path, None, whisper_model)

        # Extract just the words for display
        words = [entry['word'] for entry in transcript]
        full_text = ' '.join(words)

        logger.info(
            "[UPLOAD_VIDEO] Transcript generated, words=%d", len(words)
        )

        return jsonify({
            'success': True,
            'video_path': video_path,
            'transcript': transcript,
            'full_text': full_text,
            'word_count': len(words)
        })

    except Exception as e:
        logger.exception("[UPLOAD_VIDEO] Error processing video")
        return jsonify({'error': f'Error processing video: {str(e)}'}), 500


@app.route('/upload-video-with-txt', methods=['POST'])
def upload_video_with_txt():
    """Handle video upload with TXT transcript file."""
    log_request_context("UPLOAD_VIDEO_WITH_TXT")

    try:
        if 'video' not in request.files and 'transcript_file' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        video_file = request.files.get('video')
        txt_file = request.files.get('transcript_file')

        if not txt_file:
             return jsonify({'error': 'Transcript file is required'}), 400
        
        if video_file and video_file.filename == '':
            video_file = None

        video_path = None
        if video_file:
            if not (allowed_file(video_file.filename, ALLOWED_VIDEO_EXTENSIONS) or 
                    allowed_file(video_file.filename, ALLOWED_AUDIO_EXTENSIONS)):
                return jsonify({'error': 'Invalid file type. Video or Audio required'}), 400
            
            # Save the uploaded file
            video_filename = secure_filename(video_file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
            video_file.save(save_path)
            video_path = save_path
            
            # Check if audio and convert
            if allowed_file(video_filename, ALLOWED_AUDIO_EXTENSIONS):
                filename_no_ext = os.path.splitext(video_filename)[0]
                converted_video_filename = f"{filename_no_ext}_converted.mp4"
                converted_video_path = os.path.join(app.config['UPLOAD_FOLDER'], converted_video_filename)
                
                logger.info(f"[UPLOAD_VIDEO_WITH_TXT] Converting audio {video_filename} to video...")
                # Ensure we have the function or logic to convert.
                # Assuming convert_audio_to_video is available or we use moviepy directly here?
                # The original code had it. Let's assume it's imported or defined.
                try:
                    # We need to make sure convert_audio_to_video is defined.
                    # It seems it was used before.
                    if convert_audio_to_video(save_path, converted_video_path):
                         video_path = converted_video_path
                    else:
                         return jsonify({'error': 'Failed to convert audio file to video'}), 500
                except Exception as e:
                     logger.error(f"Failed conversion: {e}")
                     # Fallback to audio path, but downstream might fail if it expects video
                     pass

        else:
            # No video uploaded.
            pass
            

        if video_file:
            logger.info(
                "[UPLOAD_VIDEO_WITH_TXT] Using source %s (%.2f MB)",
                video_path,
                os.path.getsize(video_path) / (1024 * 1024),
            )
        else:
            logger.info("[UPLOAD_VIDEO_WITH_TXT] No main video provided. Using script-only mode.")

        transcript_text = txt_file.read().decode("utf-8", errors="replace")
        logger.info(
            "[UPLOAD_VIDEO_WITH_TXT] Transcript text length=%d",
            len(transcript_text),
        )
        lines = [line.strip() for line in transcript_text.split('\n') if line.strip()]
        logger.info(
            "[UPLOAD_VIDEO_WITH_TXT] Parsed %d non-empty lines from transcript",
            len(lines),
        )

        # Fast path: build a transcript from the provided TXT only.
        #
        # We intentionally DO NOT run Whisper/WhisperX here so the UI stays responsive.
        # Transcription + waveform alignment happens when the user clicks "Process video".
        transcript, subtitles = build_subtitle_sentences_from_lines(lines)

        if not transcript:
            return jsonify({'error': 'Transcript file produced no words'}), 400

        words = [entry['word'] for entry in transcript]
        full_text = ' '.join(words)

        logger.info(
            "[UPLOAD_VIDEO_WITH_TXT] Prepared draft transcript (words=%d, subtitles=%d). "
            "Transcription is deferred to processing.",
            len(words),
            len(subtitles),
        )

        response_data = {
            'success': True,
            'video_path': video_path,
            'transcript': transcript,
            'full_text': full_text,
            'word_count': len(words),
            'subtitles': subtitles
        }
        logger.info(
            "[UPLOAD_VIDEO_WITH_TXT] Returning draft transcript (words=%d, subtitles=%d)",
            len(words),
            len(subtitles),
        )
        return jsonify(response_data)

    except Exception as e:
        logger.exception("[UPLOAD_VIDEO_WITH_TXT] Error processing files")
        return jsonify({'error': f'Error processing files: {str(e)}'}), 500




@app.route('/upload-zip-mapping', methods=['POST'])
def upload_zip_mapping():
    """
    Handle uploading a Zip file containing clips.
    Extracts them to 'clips/' folder and returns the list of filenames.
    """
    log_request_context("UPLOAD_ZIP_MAPPING")

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not file.filename.lower().endswith('.zip'):
        return jsonify({'error': 'File must be a .zip archive'}), 400

    try:
        # Save the zip file temporarily
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, secure_filename(file.filename))
        file.save(zip_path)

        extracted_files = []
        clips_dir = os.path.join(app.root_path, 'clips')
        os.makedirs(clips_dir, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Filter for video files only to avoid junk
            file_list = [f for f in zip_ref.namelist() if not f.startswith('__MACOSX') and not f.startswith('.')]
            
            for member in file_list:
                # Basic check for video extensions
                if not any(member.lower().endswith(ext) for ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm']):
                    continue
                
                # Extract file
                # Use secure_filename on the basename to prevent directory traversal
                original_name = os.path.basename(member)
                if not original_name: 
                    continue
                    
                safe_name = secure_filename(original_name)
                target_path = os.path.join(clips_dir, safe_name)
                
                # Extract to specific path (ZipFile.extract extracts relative to CWD or path, 
                # but we want to flatten simply to clips folder usually? 
                # Or just extract normally. Let's read and write to control path perfectly.)
                with zip_ref.open(member) as source, open(target_path, "wb") as target:
                    target.write(source.read())
                
                extracted_files.append(safe_name)
                logger.info(f"[UPLOAD_ZIP] Extracted {safe_name}")

        # Cleanup zip
        os.remove(zip_path)
        os.rmdir(temp_dir)

        return jsonify({
            'success': True,
            'files': extracted_files,
            'message': f'Extracted {len(extracted_files)} clips.'
        })

    except Exception as e:
        logger.exception("[UPLOAD_ZIP] Error processing zip file")
        return jsonify({'error': f'Error processing zip: {str(e)}'}), 500


@app.route('/upload-clip', methods=['POST'])
def upload_clip():
    """Handle clip/audio file upload for highlights."""
    log_request_context("UPLOAD_CLIP")

    if 'file' not in request.files:
        logger.warning("[UPLOAD_CLIP] No 'file' in request")
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Check if it's video or audio
    is_video = allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS)
    is_audio = allowed_file(file.filename, ALLOWED_AUDIO_EXTENSIONS)

    if not (is_video or is_audio):
        return jsonify({'error': 'Invalid file type. Please upload a video or audio file.'}), 400

    try:
        filename = secure_filename(file.filename)

        # Save to appropriate folder
        if is_video:
            save_path = os.path.join('clips', filename)
        else:
            save_path = os.path.join('audio_files', filename)

        file.save(save_path)

        logger.info(
            "[UPLOAD_CLIP] Saved %s file to %s (%.2f MB)",
            "video" if is_video else "audio",
            save_path,
            os.path.getsize(save_path) / (1024 * 1024),
        )

        return jsonify({
            'success': True,
            'file_path': save_path,
            'file_type': 'video' if is_video else 'audio'
        })

    except Exception as e:
        logger.exception("[UPLOAD_CLIP] Error uploading file")
        return jsonify({'error': f'Error uploading file: {str(e)}'}), 500


@app.route('/process-video', methods=['POST'])
def process_video():
    """Process the video with highlights and generate output."""
    request_start_time = time.time()
    logger.info("=" * 80)
    logger.info("RECEIVED VIDEO PROCESSING REQUEST")
    logger.info("=" * 80)
    
    try:
        data = request.json

        video_path = data.get('video_path')
        highlights = data.get('highlights', [])
        transcript = data.get('transcript', [])
        subtitle_sentences = data.get('subtitle_sentences', [])
        aspect_ratio = data.get('aspect_ratio', '4:5')  # Default to 4:5
        render_subtitles = data.get('render_subtitles', True)
        rip_and_run = data.get('rip_and_run', False)
        
        # Strict validation removed to allow hybrid mode (Subtitles everywhere + Rip & Run)
        # if rip_and_run and render_subtitles:
        #    return jsonify({'error': 'Invalid combination: Rip & Run cannot be used with Render Subtitles (Overlay).'}), 400

        def _safe_int(value):
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        # Validate Transcript Quality & Handle Dummy Transcripts
        is_dummy_transcript = False
        if transcript and len(transcript) > 2:
            t0 = transcript[0].get('start_time', 0)
            t1 = transcript[1].get('start_time', 0)
            if abs((t1 - t0) - 0.5) < 0.001:
                is_dummy_transcript = True
                logger.info("[PROCESS_VIDEO] Detected DUMMY transcript (0.5s intervals). Forcing re-alignment.")

        backend_transcript = transcript
        base_transcript_text = None

        if is_dummy_transcript:
            backend_transcript = None # Force re-transcription
            words = [t.get('word', '') for t in transcript]
            base_transcript_text = " ".join(words)
            logger.info("[PROCESS_VIDEO] Stripping explicit indices from highlights to force phrase matching.")
            
        # Check video existence
        if not video_path or not os.path.exists(video_path):
             return jsonify({'error': 'Video file not found'}), 400

        # Sort highlights logic
        if isinstance(highlights, list) and highlights:
            indexed_highlights = list(enumerate(highlights))
            def _highlight_sort_key(item):
                original_index, highlight = item
                if not isinstance(highlight, dict):
                     return (1, original_index)
                start_idx = _safe_int(highlight.get("start_word"))
                end_idx = _safe_int(highlight.get("end_word"))
                if start_idx is None: start_idx = end_idx
                if end_idx is None: end_idx = start_idx
                if start_idx is None and end_idx is None: return (1, original_index)
                low = min(start_idx, end_idx)
                high = max(start_idx, end_idx)
                return (0, low, high, original_index)

            highlights = [h for _, h in sorted(indexed_highlights, key=_highlight_sort_key)]

        # Build Highlight Assignments
        assignments = []
        for h in highlights:
            start_w = _safe_int(h.get('start_word'))
            end_w = _safe_int(h.get('end_word'))
            
            # If dummy transcript, invalidate indices to force phrase match
            if is_dummy_transcript:
                start_w = None
                end_w = None

            assignments.append(HighlightAssignment(
                phrase=h.get('phrase', ''),
                clip_path=h.get('clip_path'),
                music_path=h.get('music_path'),
                music_volume=float(h.get('music_volume', 1.0)),
                occurrence=int(h.get('occurrence', 1) or 1),
                start_word=start_w,
                end_word=end_w
            ))

        output_filename = f"output_{Path(video_path).stem}.mp4"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        # Build Subtitle Sentences
        sentences = []
        if subtitle_sentences:
             # Use frontend sentences if we trust the transcript OR if we are re-aligning (dummy transcript)
             trust_indices = not is_dummy_transcript
             
             for s in subtitle_sentences:
                 if isinstance(s, dict):
                     sentences.append(SubtitleSentence(
                         text=s.get('text', ''),
                         phrase=s.get('text', ''),
                         occurrence=s.get('occurrence', 1),
                         start_word=_safe_int(s.get('start_word')) if trust_indices else None,
                         end_word=_safe_int(s.get('end_word')) if trust_indices else None,
                     ))
                 elif isinstance(s, str):
                      sentences.append(SubtitleSentence(text=s, phrase=s))

        # Get subtitle design based on user selection (default to 1)
        subtitle_design_number = data.get('subtitle_design_number', 1)
        subtitle_design = get_subtitle_design_by_selection(subtitle_design_number, aspect_ratio)

        config = ProjectConfig(
            main_video_path=video_path,
            highlight_assignments=assignments,
            output_path=output_path,
            preserve_audio=data.get('preserve_audio', True),
            subtitle_sentences=sentences,
            aspect_ratio=aspect_ratio,
            render_subtitles=render_subtitles,
            rip_and_run=rip_and_run,
            subtitle_design=subtitle_design,
            transcript=backend_transcript,
            transcript_text=base_transcript_text
        )
        # Render the project with the existing transcript
        logger.info("[REQUEST] Starting render_project...")
        render_start = time.time()
        render_project(config)
        render_duration = time.time() - render_start
        logger.info(f"[REQUEST] ✓ render_project completed in {render_duration:.2f}s")

        # Create project JSON file
        logger.info("[REQUEST] Creating project JSON file...")
        project_data = create_project_json(
            video_path=video_path,
            highlights=highlights,
            transcript=transcript,
            subtitle_sentences=subtitle_sentences,
            aspect_ratio=aspect_ratio,
            output_filename=output_filename,
            output_path=output_path
        )
        
        # Save project JSON locally
        project_filename = f"project_{Path(video_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        project_path = os.path.join(app.config['OUTPUT_FOLDER'], project_filename)
        with open(project_path, 'w', encoding='utf-8') as f:
            json.dump(project_data, f, indent=2, ensure_ascii=False)
        
        # Upload to S3
        s3_video_url = None
        s3_project_url = None
        try:
            # Create S3 keys with timestamp for organization
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            video_s3_key = f"videos/{timestamp}_{output_filename}"
            project_s3_key = f"projects/{timestamp}_{project_filename}"
            
            # Upload video to S3
            logger.info(f"[REQUEST] Uploading video to S3: {video_s3_key}")
            s3_start = time.time()
            s3_video_url = upload_file_to_s3(output_path, video_s3_key)
            s3_duration = time.time() - s3_start
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"[REQUEST] ✓ Video uploaded to S3 in {s3_duration:.2f}s ({file_size_mb:.2f} MB) - {s3_video_url}")
            
            # Upload project JSON to S3
            logger.info(f"[REQUEST] Uploading project file to S3: {project_s3_key}")
            s3_start = time.time()
            s3_project_url = upload_file_to_s3(project_path, project_s3_key)
            s3_duration = time.time() - s3_start
            logger.info(f"[REQUEST] ✓ Project file uploaded to S3 in {s3_duration:.2f}s - {s3_project_url}")
            
        except Exception as s3_error:
            logger.warning(f"[REQUEST] S3 upload failed: {s3_error}")
            # Continue even if S3 upload fails - local files are still available

        total_request_time = time.time() - request_start_time
        logger.info("=" * 80)
        logger.info(f"[REQUEST] ✓ REQUEST COMPLETED in {total_request_time:.2f}s ({total_request_time/60:.2f} minutes)")
        logger.info("=" * 80)

        return jsonify({
            'success': True,
            'output_path': output_path,
            'output_filename': output_filename,
            'project_filename': project_filename,
            'project_path': project_path,
            's3_video_url': s3_video_url,
            's3_project_url': s3_project_url,
            'message': 'Video processed successfully!' + (' (Uploaded to S3)' if s3_video_url else ' (S3 upload failed)')
        })

    except Exception:
        logger.exception("[REQUEST] Unhandled error while processing video")
        return jsonify({'error': 'Error processing video'}), 500


@app.route('/download/<filename>')
def download_file(filename):
    output_folder = app.config['OUTPUT_FOLDER']
    # Only allow .mp4 files
    if not filename.lower().endswith('.mp4'):
        return jsonify({'error': 'File not found'}), 404
    file_path = os.path.join(output_folder, filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    # Stream the file as an attachment
    resp = send_from_directory(
        output_folder,
        filename,
        as_attachment=True,
        mimetype='video/mp4',
    )

    # Disable caching so every click always fetches the latest file
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"

    return resp

@app.route('/video/<filename>')
def view_video(filename):
    """Serve the processed video for preview (not as download)."""
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(file_path):
        resp = send_file(file_path, mimetype='video/mp4')
        # Disable caching so the preview ALWAYS matches the latest render
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp
    return jsonify({'error': 'File not found'}), 404


@app.route('/clips/<filename>')
def serve_clip(filename):
    """Serve a clip file."""
    return send_from_directory('clips', filename)

@app.route('/audio_files/<filename>')
def serve_audio(filename):
    """Serve an audio file."""
    return send_from_directory('audio_files', filename)


@app.route('/api/assets', methods=['GET'])
def list_assets():
    """Return JSON of all video/audio assets with sizes and modification times."""
    videos = []
    audio = []

    # Videos
    if os.path.exists('clips'):
        for f in os.listdir('clips'):
            path = os.path.join('clips', f)
            if os.path.isfile(path) and f.lower().endswith(tuple(ALLOWED_VIDEO_EXTENSIONS)):
                try:
                    mtime = os.path.getmtime(path)
                    videos.append({
                        'name': f,
                        'path': f'clips/{f}',
                        'size': os.path.getsize(path),
                        'date': datetime.fromtimestamp(mtime).strftime('%d %b %Y')
                    })
                except OSError:
                    pass

    # Audio
    if os.path.exists('audio_files'):
        for f in os.listdir('audio_files'):
            path = os.path.join('audio_files', f)
            if os.path.isfile(path) and f.lower().endswith(tuple(ALLOWED_AUDIO_EXTENSIONS)):
                try:
                    mtime = os.path.getmtime(path)
                    audio.append({
                        'name': f,
                        'path': f'audio_files/{f}',
                        'size': os.path.getsize(path),
                        'date': datetime.fromtimestamp(mtime).strftime('%d %b %Y')
                    })
                except OSError:
                    pass
    
    return jsonify({'videos': videos, 'audio': audio})


@app.route('/api/assets/<asset_type>/<filename>', methods=['DELETE'])
def delete_asset(asset_type, filename):
    """Delete an asset file."""
    try:
        # Validate directory based on type
        if asset_type == 'video':
            directory = 'clips'
        elif asset_type == 'audio':
            directory = 'audio_files'
        else:
            return jsonify({'error': 'Invalid asset type'}), 400
            
        # Security check for filename
        filename = secure_filename(filename)
        if not filename:
            return jsonify({'error': 'Invalid filename'}), 400
            
        file_path = os.path.join(directory, filename)
        
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted asset: {file_path}")
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'File not found'}), 404
            
    except Exception as e:
        logger.error(f"Error deleting asset: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/list-projects', methods=['GET'])
def list_projects():
    """List all project files from S3."""
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET_NAME,
            Prefix='projects/'
        )
        
        projects = []
        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['Key'].endswith('.json'):
                    # Extract project info from key
                    filename = obj['Key'].split('/')[-1]
                    projects.append({
                        'key': obj['Key'],
                        'filename': filename,
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat(),
                        's3_url': f"https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/{obj['Key']}"
                    })
        
        # Sort by last modified (newest first)
        projects.sort(key=lambda x: x['last_modified'], reverse=True)
        
        return jsonify({
            'success': True,
            'projects': projects
        })
    except ClientError as e:
        print(f"Error listing projects from S3: {e}")
        return jsonify({'error': f'Error listing projects: {str(e)}'}), 500


@app.route('/load-project', methods=['POST'])
def load_project():
    """Load a project from S3 and return its data."""
    try:
        data = request.json
        project_key = data.get('project_key') or data.get('s3_url')
        
        if not project_key:
            return jsonify({'error': 'Project key or S3 URL is required'}), 400
        
        # Extract key from URL if full URL is provided
        if project_key.startswith('http'):
            # Extract key from URL: https://bucket.s3.region.amazonaws.com/projects/filename.json
            project_key = '/'.join(project_key.split('/')[3:])
        
        # Download project file from S3
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=project_key)
        project_data = json.loads(response['Body'].read().decode('utf-8'))
        
        return jsonify({
            'success': True,
            'project': project_data
        })
    except ClientError as e:
        print(f"Error loading project from S3: {e}")
        return jsonify({'error': f'Error loading project: {str(e)}'}), 500
    except Exception as e:
        print(f"Error parsing project: {e}")
        return jsonify({'error': f'Error parsing project: {str(e)}'}), 500


@app.route('/save-project', methods=['POST'])
def save_project():
    """Save a project to S3 without processing the video."""
    request_start = time.time()
    logger.info("[SAVE PROJECT] Received save project request")
    
    try:
        data = request.json

        video_path = data.get('video_path')
        highlights = data.get('highlights', [])
        transcript = data.get('transcript', [])
        subtitle_sentences = data.get('subtitle_sentences', [])
        aspect_ratio = data.get('aspect_ratio', '4:5')
        project_name = data.get('project_name', None)  # Optional custom name
        
        logger.info(f"[SAVE PROJECT] Highlights: {len(highlights)}, Transcript words: {len(transcript)}")

        # Validate aspect ratio
        if aspect_ratio not in ['4:5', '9:16']:
            aspect_ratio = '4:5'

        if not video_path:
            return jsonify({'error': 'Video path is required'}), 400

        # Create project JSON file
        if project_name:
            # Use custom name if provided
            project_filename = f"{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            project_filename = f"project_{Path(video_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        project_data = create_project_json(
            video_path=video_path,
            highlights=highlights,
            transcript=transcript,
            subtitle_sentences=subtitle_sentences,
            aspect_ratio=aspect_ratio,
            output_filename=None,  # No output yet
            output_path=None  # No output yet
        )
        
        # Mark as draft/unsaved if not processed
        project_data['project_info']['status'] = 'draft'
        project_data['project_info']['saved_at'] = datetime.now().isoformat()
        
        # Save project JSON locally first
        project_path = os.path.join(app.config['OUTPUT_FOLDER'], project_filename)
        with open(project_path, 'w', encoding='utf-8') as f:
            json.dump(project_data, f, indent=2, ensure_ascii=False)
        
        # Upload to S3
        s3_project_url = None
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            project_s3_key = f"projects/{timestamp}_{project_filename}"
            
            logger.info(f"[SAVE PROJECT] Uploading project to S3: {project_s3_key}")
            s3_start = time.time()
            s3_project_url = upload_file_to_s3(project_path, project_s3_key)
            s3_duration = time.time() - s3_start
            logger.info(f"[SAVE PROJECT] ✓ Project uploaded to S3 in {s3_duration:.2f}s - {s3_project_url}")
            
        except Exception as s3_error:
            logger.error(f"[SAVE PROJECT] S3 upload failed: {s3_error}")
            return jsonify({
                'error': f'S3 upload failed: {str(s3_error)}',
                'local_path': project_path
            }), 500

        total_time = time.time() - request_start
        logger.info(f"[SAVE PROJECT] ✓ Save completed in {total_time:.2f}s")
        
        return jsonify({
            'success': True,
            'project_filename': project_filename,
            'project_path': project_path,
            's3_project_url': s3_project_url,
            'message': 'Project saved successfully to S3!'
        })

    except Exception as e:
        logger.exception("[SAVE PROJECT] Error while saving project")
        return jsonify({'error': f'Error saving project: {str(e)}'}), 500


if __name__ == '__main__':
    # Disable reloader to prevent server restarts during video processing
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting app on port {port}...")
    app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)
