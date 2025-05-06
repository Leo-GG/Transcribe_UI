"""
Audio processing utilities for the Audio Transcription App.
"""

import os
import tempfile
from pydub import AudioSegment
import streamlit as st


def convert_audio_to_wav(input_file):
    """
    Convert an audio file to WAV format for processing.
    
    Args:
        input_file (str): Path to the input audio file
        
    Returns:
        str: Path to the converted WAV file
    """
    try:
        audio = AudioSegment.from_file(input_file)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        audio.export(temp_file.name, format='wav')
        return temp_file.name
    except Exception as e:
        st.error(f"Error converting audio: {e}")
        return input_file


def save_temp_audio_file(uploaded_file):
    """
    Save an uploaded file to a temporary location.
    
    Args:
        uploaded_file (UploadedFile): The uploaded file from Streamlit
        
    Returns:
        str: Path to the saved temporary file
    """
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.getvalue())
    temp_file_path = temp_file.name
    temp_file.close()
    
    return temp_file_path


def cleanup_temp_files(file_paths):
    """
    Clean up temporary files after processing.
    
    Args:
        file_paths (list): List of file paths to clean up
    """
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            st.warning(f"Warning: Could not delete temporary file {file_path}: {e}")


def get_audio_duration(file_path):
    """
    Get the duration of an audio file in seconds.
    
    Args:
        file_path (str): Path to the audio file
        
    Returns:
        float: Duration of the audio file in seconds
    """
    try:
        audio = AudioSegment.from_file(file_path)
        return len(audio) / 1000.0  # Convert milliseconds to seconds
    except Exception as e:
        st.error(f"Error getting audio duration: {e}")
        return 0.0


def get_audio_metadata(file_path):
    """
    Get metadata for an audio file.
    
    Args:
        file_path (str): Path to the audio file
        
    Returns:
        dict: Dictionary containing audio metadata
    """
    try:
        audio = AudioSegment.from_file(file_path)
        return {
            "channels": audio.channels,
            "sample_width": audio.sample_width,
            "frame_rate": audio.frame_rate,
            "frame_width": audio.frame_width,
            "duration_seconds": len(audio) / 1000.0
        }
    except Exception as e:
        st.error(f"Error getting audio metadata: {e}")
        return {}
