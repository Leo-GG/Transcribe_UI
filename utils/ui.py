"""
UI components and helpers for the Audio Transcription App.
"""

import streamlit as st
import os
import time


def setup_page_config():
    """
    Set up the Streamlit page configuration.
    """
    st.set_page_config(
        page_title="Audio Transcription App",
        page_icon="🎙️",
        layout="wide"
    )


def render_header():
    """
    Render the app header and description.
    """
    st.title("🎙️ Audio Transcription App")
    st.markdown("""
    This app allows you to transcribe audio files to text using a local Whisper model.
    Upload one or more audio files and select the language for transcription.
    """)


def render_sidebar(available_languages):
    """
    Render the sidebar with model settings.
    
    Args:
        available_languages (list): List of available languages
        
    Returns:
        tuple: (model_size, language, device, enable_diarization)
    """
    with st.sidebar:
        st.header("Model Settings")
        
        model_size = st.selectbox(
            "Select Model Size",
            ["tiny", "base", "small", "medium", "large-v2"],
            index=1,
            help="Larger models are more accurate but slower and require more memory"
        )
        
        language = st.selectbox(
            "Select Language",
            available_languages,
            index=0,
            help="Select the primary language of the audio"
        )
        
        device = st.radio(
            "Compute Device",
            ["cpu", "cuda"],
            index=0,
            help="CUDA (GPU) is faster but requires compatible NVIDIA GPU"
        )
        
        st.markdown("---")
        st.header("Speaker Identification")
        
        enable_diarization = st.checkbox(
            "Enable Speaker Identification", 
            value=True,
            help="Identify different speakers in the audio (may increase processing time)"
        )
        
        if enable_diarization:
            st.info("Speaker identification uses local machine learning algorithms to detect different speakers. No data is sent to external servers.")
            
            st.markdown("""💡 **Note:** The speaker identification is based on voice characteristics 
            and may not be 100% accurate, especially for short audio segments or similar voices.""")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This app uses the faster-whisper implementation of OpenAI's Whisper model to 
        transcribe audio locally without sending data to external servers.
        """)
        
        return model_size, language, device, enable_diarization


def render_file_uploader():
    """
    Render the file uploader component.
    
    Returns:
        list: List of uploaded files
    """
    uploaded_files = st.file_uploader(
        "Upload audio files (MP3, WAV, M4A, etc.)",
        type=["mp3", "wav", "m4a", "ogg", "flac", "aac"],
        accept_multiple_files=True
    )
    
    return uploaded_files


def render_transcription_section(uploaded_file, transcribe_callback):
    """
    Render the transcription section for a single file.
    
    Args:
        uploaded_file (UploadedFile): The uploaded file
        transcribe_callback (callable): Callback function for transcription
        
    Returns:
        bool: Whether transcription was performed
    """
    # Display file info
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### 📄 {uploaded_file.name}")
    
    # Transcribe button
    transcribe_button = st.button(
        f"Transcribe {uploaded_file.name}", 
        key=f"btn_{uploaded_file.name}"
    )
    
    if transcribe_button:
        with st.spinner(f"Transcribing {uploaded_file.name}..."):
            start_time = time.time()
            transcription, detected_language, segments_with_speakers = transcribe_callback()
            end_time = time.time()
        
        # Display transcription
        st.markdown("#### Transcription:")
        
        # Check if we have speaker information
        has_speaker_info = segments_with_speakers is not None
        
        if has_speaker_info:
            # Create a more visually appealing display for speaker-labeled transcripts
            st.markdown("""<style>
            .speaker-label { font-weight: bold; color: #1E88E5; }
            .timestamp { color: #757575; font-size: 0.9em; }
            .speaker-text { margin-bottom: 1em; }
            </style>""", unsafe_allow_html=True)
            
            for segment in segments_with_speakers:
                speaker = segment["speaker"]
                text = segment["text"]
                start_time = segment["start"]
                end_time = segment["end"]
                
                # Format timestamp as MM:SS
                start_str = f"{int(start_time // 60):02d}:{int(start_time % 60):02d}"
                end_str = f"{int(end_time // 60):02d}:{int(end_time % 60):02d}"
                
                st.markdown(f"""<div>
                <span class='timestamp'>[{start_str} → {end_str}]</span> 
                <span class='speaker-label'>{speaker}:</span> {text}
                </div>
                <div class='speaker-text'></div>""", unsafe_allow_html=True)
        else:
            # Standard display for transcription without speaker labels
            st.text_area(
                "Transcribed Text",
                transcription,
                height=200,
                key=f"transcript_{uploaded_file.name}"
            )
        
        # Display metadata
        st.markdown("#### Metadata:")
        col1, col2, col3 = st.columns(3)
        col1.metric("Processing Time", f"{end_time - start_time:.2f} seconds")
        col2.metric("Detected Language", detected_language)
        col3.metric("File Size", f"{len(uploaded_file.getvalue()) / (1024*1024):.2f} MB")
        
        # Download button
        st.download_button(
            label="Download Transcription",
            data=transcription,
            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcription.txt",
            mime="text/plain",
            key=f"download_{uploaded_file.name}"
        )
        
        st.markdown("---")
        return True
    
    return False


def display_error(message):
    """
    Display an error message.
    
    Args:
        message (str): Error message to display
    """
    st.error(message)


def display_info(message):
    """
    Display an info message.
    
    Args:
        message (str): Info message to display
    """
    st.info(message)


def display_success(message):
    """
    Display a success message.
    
    Args:
        message (str): Success message to display
    """
    st.success(message)


def display_warning(message):
    """
    Display a warning message.
    
    Args:
        message (str): Warning message to display
    """
    st.warning(message)
