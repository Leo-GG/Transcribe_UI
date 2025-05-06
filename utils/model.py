"""
Model loading and transcription utilities for the Audio Transcription App.
"""

import streamlit as st
from faster_whisper import WhisperModel
from utils.diarization import perform_speaker_diarization, estimate_num_speakers, merge_transcription_with_speakers, format_transcript_with_speakers


@st.cache_resource
def load_model(model_size, device):
    """
    Load the Whisper model with caching.
    
    Args:
        model_size (str): Size of the model to load ('tiny', 'base', 'small', 'medium', 'large-v2')
        device (str): Device to use for computation ('cpu' or 'cuda')
        
    Returns:
        WhisperModel: Loaded model instance
    """
    st.info(f"Loading {model_size} model... This may take a moment.")
    try:
        compute_type = "int8"
        # Use float16 for GPU to improve performance
        if device == "cuda":
            compute_type = "float16"
        
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("If using CUDA, ensure you have a compatible GPU and proper drivers installed.")
        return None


def transcribe_audio(audio_path, model, selected_language, enable_diarization=False):
    """
    Transcribe audio using the Whisper model with optional speaker diarization.
    
    Args:
        audio_path (str): Path to the audio file
        model (WhisperModel): Loaded Whisper model
        selected_language (str): Language for transcription ('English', 'Spanish', etc.)
        enable_diarization (bool): Whether to enable speaker diarization
        
    Returns:
        tuple: (transcription_text, detected_language, segments_with_speakers)
    """
    # Map UI language selection to language codes
    language_map = {
        "English": "en",
        "Spanish": "es",
        # Add more languages as needed
    }
    
    lang_code = language_map.get(selected_language, "en")
    
    try:
        # Run transcription
        segments, info = model.transcribe(
            audio_path, 
            language=lang_code,
            task="transcribe",
            beam_size=5
        )
        
        # Convert generator to list for multiple use
        segments_list = list(segments)
        
        # Perform speaker diarization if enabled
        if enable_diarization:
            with st.spinner("Identifying speakers..."):
                # Estimate number of speakers
                num_speakers = estimate_num_speakers(audio_path)
                st.info(f"Detected approximately {num_speakers} speakers")
                
                # Perform diarization
                speaker_segments = perform_speaker_diarization(audio_path, num_speakers)
                
                # Merge transcription with speaker information
                segments_with_speakers = merge_transcription_with_speakers(segments_list, speaker_segments)
                
                # Format transcript with speaker information
                transcription_text = format_transcript_with_speakers(segments_with_speakers)
        else:
            # Standard transcription without diarization
            transcription_text = "".join([segment.text for segment in segments_list])
            segments_with_speakers = None
        
        return transcription_text, info.language, segments_with_speakers
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return f"Transcription failed: {str(e)}", "unknown", None


def get_available_languages():
    """
    Get a list of available languages for transcription.
    
    Returns:
        list: List of available languages
    """
    # This could be expanded in the future
    return ["English", "Spanish"]


def get_model_info(model_size):
    """
    Get information about a specific model size.
    
    Args:
        model_size (str): Size of the model ('tiny', 'base', 'small', 'medium', 'large-v2')
        
    Returns:
        dict: Dictionary containing model information
    """
    model_info = {
        "tiny": {
            "parameters": "39M",
            "relative_speed": "Very Fast",
            "memory_usage": "Low",
            "accuracy": "Basic"
        },
        "base": {
            "parameters": "74M",
            "relative_speed": "Fast",
            "memory_usage": "Low",
            "accuracy": "Good"
        },
        "small": {
            "parameters": "244M",
            "relative_speed": "Medium",
            "memory_usage": "Medium",
            "accuracy": "Better"
        },
        "medium": {
            "parameters": "769M",
            "relative_speed": "Slow",
            "memory_usage": "High",
            "accuracy": "Great"
        },
        "large-v2": {
            "parameters": "1550M",
            "relative_speed": "Very Slow",
            "memory_usage": "Very High",
            "accuracy": "Best"
        }
    }
    
    return model_info.get(model_size, {})
