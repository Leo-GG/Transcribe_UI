import os
import time
import streamlit as st
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Import utility modules
from utils.audio import convert_audio_to_wav, save_temp_audio_file, cleanup_temp_files, get_audio_metadata
from utils.model import load_model, transcribe_audio, get_available_languages
from utils.ui import setup_page_config, render_header, render_sidebar, render_file_uploader, render_transcription_section

# Set up page configuration
setup_page_config()

# Render app header
render_header()

# Render sidebar and get selected options
available_languages = get_available_languages()
model_size, language, device, enable_diarization = render_sidebar(available_languages)

# Main app functionality
def process_file(uploaded_file, model):
    """Process a single uploaded file for transcription.
    
    Args:
        uploaded_file: The uploaded file from Streamlit
        model: The loaded Whisper model
        
    Returns:
        tuple: (temp_file_path, wav_file_path) - paths to be cleaned up later
    """
    # Save uploaded file to temporary location
    temp_file_path = save_temp_audio_file(uploaded_file)
    
    # Convert to WAV if needed
    wav_file_path = convert_audio_to_wav(temp_file_path)
    
    return temp_file_path, wav_file_path

def main():
    """Main application function."""
    # Load the model
    model = load_model(model_size, device)
    
    # Exit if model failed to load
    if model is None:
        st.error("Failed to load the transcription model. Please check your settings and try again.")
        return
    
    # File uploader
    uploaded_files = render_file_uploader()
    
    if uploaded_files:
        st.markdown("---")
        st.subheader("Transcription Results")
        
        # Process each uploaded file
        for uploaded_file in uploaded_files:
            # Process file paths
            temp_file_path, wav_file_path = process_file(uploaded_file, model)
            
            # Define the transcription callback
            def perform_transcription():
                return transcribe_audio(wav_file_path, model, language, enable_diarization)
            
            # Render the transcription section
            transcription_performed = render_transcription_section(
                uploaded_file, 
                perform_transcription
            )
            
            # Clean up temporary files
            cleanup_temp_files([temp_file_path, wav_file_path])

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.info("Please refresh the page and try again.")
        # Log the error (in a real app, you might want to log to a file)
        import traceback
        traceback.print_exc()
