"""
Speaker diarization utilities for the Audio Transcription App.
"""

import numpy as np
import librosa
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import streamlit as st
import os
import tempfile
import warnings
import soundfile as sf
import traceback
import shutil  # For checking if ffmpeg is available

# Try to import pydub for audio conversion fallback
try:
    import pydub
except ImportError:
    # We'll handle this later if needed
    pass

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


def extract_audio_features(audio_path, n_mfcc=20, n_mels=128):
    """
    Extract MFCC features from audio for speaker diarization.
    
    Args:
        audio_path (str): Path to the audio file
        n_mfcc (int): Number of MFCC coefficients to extract
        n_mels (int): Number of Mel bands to use
        
    Returns:
        tuple: (features, timestamps, audio_duration)
    """
    try:
        # Check if file exists
        if not os.path.exists(audio_path):
            st.error(f"Audio file not found: {audio_path}")
            return None, None, 0
            
        # Create a temporary file if needed for certain formats
        temp_file = None
        file_ext = os.path.splitext(audio_path)[1].lower()
        
        try:
            # First try using soundfile which is more reliable
            try:
                st.info(f"Attempting to load audio with soundfile: {audio_path}")
                # Try to load with soundfile first (handles WAV files well)
                audio_data, samplerate = sf.read(audio_path, dtype="float32")
                
                # Create a temporary WAV file that librosa can process
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp:
                    temp_file = temp.name
                
                # Save as PCM WAV format which is well-supported
                sf.write(temp_file, audio_data, samplerate, 'PCM_16')
                
                # Now load with librosa from the temp file
                y, sr = librosa.load(temp_file, sr=None)
                st.success("Successfully loaded audio with soundfile fallback")
                
            except Exception as sf_error:
                st.warning(f"Soundfile loading failed: {sf_error}. Trying librosa directly...")
                
                # Try direct librosa loading as fallback
                try:
                    y, sr = librosa.load(audio_path, sr=None, res_type='kaiser_fast')
                    st.success("Successfully loaded audio with librosa directly")
                    
                except Exception as librosa_error:
                    # If both methods fail, try with ffmpeg conversion
                    st.warning(f"Librosa loading failed: {librosa_error}. Trying ffmpeg conversion...")
                    
                    # Create a temporary WAV file
                    if not temp_file:
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp:
                            temp_file = temp.name
                    
                    # Check if ffmpeg is available
                    import shutil
                    ffmpeg_available = shutil.which('ffmpeg') is not None
                    
                    if ffmpeg_available:
                        # Use ffmpeg to convert to WAV
                        import subprocess
                        try:
                            st.info(f"Converting audio with ffmpeg to {temp_file}")
                            result = subprocess.run(
                                ['ffmpeg', '-i', audio_path, '-ar', '44100', '-ac', '1', temp_file],
                                check=True, capture_output=True, text=True
                            )
                            st.info(f"ffmpeg conversion output: {result.stdout}")
                            
                            # Now try to load the converted file
                            y, sr = librosa.load(temp_file, sr=None)
                            st.success("Successfully loaded audio with ffmpeg conversion")
                        except Exception as ffmpeg_error:
                            st.error(f"ffmpeg conversion failed: {ffmpeg_error}")
                            if hasattr(ffmpeg_error, 'stderr'):
                                st.error(f"ffmpeg stderr: {ffmpeg_error.stderr}")
                            raise
                    else:
                        # If ffmpeg is not available, try a pure Python approach
                        st.warning("ffmpeg not found. Trying alternative approach...")
                        try:
                            # Try to read the file directly with a different method
                            from pydub import AudioSegment
                            audio = AudioSegment.from_file(audio_path)
                            audio.export(temp_file, format="wav")
                            
                            # Now try to load the converted file
                            y, sr = librosa.load(temp_file, sr=None)
                            st.success("Successfully loaded audio with pydub conversion")
                        except ImportError:
                            st.error("pydub not installed. Please install with: pip install pydub")
                            # Continue with the error handling below
                            raise Exception("Neither ffmpeg nor pydub is available for audio conversion")
            
            except Exception as ffmpeg_error:
                st.error(f"All audio loading methods failed. Last error (ffmpeg): {ffmpeg_error}")
                if hasattr(ffmpeg_error, 'stderr'):
                    st.error(f"ffmpeg stderr: {ffmpeg_error.stderr}")
                return None, None, 0
            
            # Verify audio data was loaded
            if y.size == 0 or sr == 0:
                st.error("Audio data is empty or sample rate is zero")
                return None, None, 0
                
            # Extract features
            hop_length = int(sr * 0.01)  # 10ms hop
            frame_length = int(sr * 0.05)  # 50ms window
            
            # Extract MFCCs with error handling
            st.info(f"Extracting MFCCs with sr={sr}, hop_length={hop_length}, frame_length={frame_length}")
            mfccs = librosa.feature.mfcc(
                y=y, 
                sr=sr, 
                n_mfcc=n_mfcc,
                n_mels=n_mels,
                hop_length=hop_length,
                n_fft=frame_length
            )
            
            # Verify MFCC extraction worked
            if mfccs.size == 0:
                st.error("MFCC extraction produced empty features")
                return None, None, 0
            
            st.info(f"Successfully extracted MFCCs with shape {mfccs.shape}")
            
            # Add delta features
            delta_mfccs = librosa.feature.delta(mfccs)
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            
            # Combine features
            features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
            features = features.T  # Transpose to get time as first dimension
            
            # Calculate timestamps for each frame
            timestamps = librosa.frames_to_time(
                np.arange(features.shape[0]), 
                sr=sr, 
                hop_length=hop_length
            )
            
            st.success(f"Audio feature extraction complete: {features.shape} features extracted")
            return features, timestamps, len(y) / sr
            
        finally:
            # Clean up temporary file if created
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                    st.info(f"Cleaned up temporary file: {temp_file}")
                except Exception as cleanup_error:
                    st.warning(f"Failed to clean up temporary file: {cleanup_error}")
    
    except Exception as e:
        st.error(f"Error extracting audio features: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        
        # Provide more helpful error messages based on the exception type
        error_text = str(e)
        traceback_text = str(traceback.format_exc())
        
        if "NoBackendError" in error_text or "NoBackendError" in traceback_text:
            st.error("""
            NoBackendError: librosa requires audio backends to process files.
            Please install one of the following:
            1. Install ffmpeg: https://ffmpeg.org/download.html
            2. Install PyAudio: pip install PyAudio
            3. Install SoundFile: pip install SoundFile
            """)
        elif "Format not recognised" in error_text or "Format not recognised" in traceback_text:
            st.error("""
            Format not recognized: The audio format is not supported.
            Try converting your audio to WAV format using an online converter
            or install ffmpeg: https://ffmpeg.org/download.html
            """)
        elif "WinError 2" in error_text or "WinError 2" in traceback_text:
            st.error("""
            ffmpeg not found: The system cannot find ffmpeg.
            Please install ffmpeg and add it to your PATH:
            1. Download from: https://ffmpeg.org/download.html
            2. Add the bin directory to your system PATH
            3. Restart your application
            
            Alternatively, install pydub for audio conversion:
            pip install pydub
            """)
            
        return None, None, 0


def perform_speaker_diarization(audio_path, num_speakers=2, min_segment_duration=1.0):
    """
    Perform speaker diarization on an audio file.
    
    Args:
        audio_path (str): Path to the audio file
        num_speakers (int): Number of speakers to identify (default 2)
        min_segment_duration (float): Minimum duration for a speaker segment in seconds
        
    Returns:
        list: List of speaker segments with start time, end time, and speaker ID
    """
    try:
        # Extract features
        features, timestamps, audio_duration = extract_audio_features(audio_path)
        
        if features is None or len(features) == 0:
            st.error("Failed to extract audio features for diarization. Returning empty speaker segments.")
            return []
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Perform clustering
        clustering = SpectralClustering(
            n_clusters=num_speakers,
            assign_labels="discretize",
            random_state=42,
            affinity="nearest_neighbors"
        )
        
        labels = clustering.fit_predict(scaled_features)
        
        # Convert frame-level labels to segments
        segments = []
        current_speaker = labels[0]
        segment_start = timestamps[0]
        
        for i in range(1, len(labels)):
            if labels[i] != current_speaker:
                # End the current segment
                segment_end = timestamps[i]
                
                # Only add segments longer than min_segment_duration
                if segment_end - segment_start >= min_segment_duration:
                    segments.append({
                        "start": segment_start,
                        "end": segment_end,
                        "speaker": f"SPEAKER_{current_speaker + 1}"
                    })
                
                # Start a new segment
                current_speaker = labels[i]
                segment_start = timestamps[i]
        
        # Add the final segment
        if timestamps[-1] - segment_start >= min_segment_duration:
            segments.append({
                "start": segment_start,
                "end": timestamps[-1],
                "speaker": f"SPEAKER_{current_speaker + 1}"
            })
        
        return segments
    
    except Exception as e:
        st.error(f"Error in speaker diarization: {e}")
        return []


def estimate_num_speakers(audio_path, max_speakers=5):
    """
    Estimate the number of speakers in an audio file.
    
    Args:
        audio_path (str): Path to the audio file
        max_speakers (int): Maximum number of speakers to consider
        
    Returns:
        int: Estimated number of speakers
    """
    try:
        # Extract features
        features, _, _ = extract_audio_features(audio_path)
        
        if features is None or len(features) == 0:
            return 2  # Default to 2 speakers
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Sample features to speed up computation
        if len(scaled_features) > 1000:
            indices = np.random.choice(len(scaled_features), 1000, replace=False)
            sampled_features = scaled_features[indices]
        else:
            sampled_features = scaled_features
        
        # Calculate silhouette scores for different numbers of clusters
        best_score = -1
        best_n_speakers = 2
        
        for n_speakers in range(2, min(max_speakers + 1, 6)):
            # Skip if we have too few samples
            if len(sampled_features) <= n_speakers:
                continue
                
            clustering = SpectralClustering(
                n_clusters=n_speakers,
                assign_labels="discretize",
                random_state=42,
                affinity="nearest_neighbors"
            )
            
            labels = clustering.fit_predict(sampled_features)
            
            # Calculate cluster quality
            cluster_centers = []
            for i in range(n_speakers):
                cluster_samples = sampled_features[labels == i]
                if len(cluster_samples) > 0:
                    cluster_centers.append(np.mean(cluster_samples, axis=0))
            
            # Calculate average distance between cluster centers
            if len(cluster_centers) > 1:
                distances = cdist(cluster_centers, cluster_centers)
                np.fill_diagonal(distances, np.inf)
                avg_min_distance = np.mean(np.min(distances, axis=1))
                
                # Higher score is better
                score = avg_min_distance
                
                if score > best_score:
                    best_score = score
                    best_n_speakers = n_speakers
        
        return best_n_speakers
    
    except Exception as e:
        st.error(f"Error estimating number of speakers: {e}")
        return 2  # Default to 2 speakers


def merge_transcription_with_speakers(segments, speaker_segments):
    """
    Merge transcription segments with speaker information.
    
    Args:
        segments (list): List of transcription segments from Whisper
        speaker_segments (list): List of speaker segments from diarization
        
    Returns:
        list: List of segments with added speaker information
    """
    result = []
    
    for segment in segments:
        start = segment.start
        end = segment.end
        text = segment.text
        
        # Find the speaker for this segment
        speaker = "UNKNOWN"
        max_overlap = 0
        
        for spk_segment in speaker_segments:
            spk_start = spk_segment["start"]
            spk_end = spk_segment["end"]
            
            # Calculate overlap
            overlap_start = max(start, spk_start)
            overlap_end = min(end, spk_end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > max_overlap:
                max_overlap = overlap
                speaker = spk_segment["speaker"]
        
        # Add speaker information to the segment
        result.append({
            "start": start,
            "end": end,
            "text": text,
            "speaker": speaker
        })
    
    return result


def format_transcript_with_speakers(segments_with_speakers):
    """
    Format the transcript with speaker information.
    
    Args:
        segments_with_speakers (list): List of segments with speaker information
        
    Returns:
        str: Formatted transcript with speaker labels
    """
    result = []
    
    for segment in segments_with_speakers:
        start_time = format_timestamp(segment["start"])
        end_time = format_timestamp(segment["end"])
        speaker = segment["speaker"]
        text = segment["text"].strip()
        
        result.append(f"[{start_time} -> {end_time}] {speaker}: {text}")
    
    return "\n".join(result)


def format_timestamp(seconds):
    """
    Format seconds as MM:SS.
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted time string
    """
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"
