# Audio Transcription App

A Streamlit web application for transcribing audio files to text using a local Whisper model. This app works completely offline without requiring external servers.

## Features

- Upload one or multiple audio files for transcription
- Support for various audio formats (MP3, WAV, M4A, OGG, FLAC, AAC)
- Speaker diarization to identify different speakers in the audio
- Language selection (English or Spanish)
- Multiple model size options to balance speed and accuracy
- Download transcription results as text files
- Completely local processing - no data sent to external servers
- Detailed metadata for each transcription (processing time, detected language)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Advanced Usage](#advanced-usage)
- [Speaker Diarization](#speaker-diarization)
- [Model Information](#model-information)
- [Requirements](#requirements)
- [Performance Considerations](#performance-considerations)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone this repository or download the files:

```bash
git clone https://github.com/yourusername/Transcribe_UI.git
cd Transcribe_UI
```

2. Create a virtual environment (recommended):

```bash
# Using venv
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Install FFmpeg (required for audio processing):
   - Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html) and add to PATH
   - Linux: `sudo apt install ffmpeg`
   - macOS: `brew install ffmpeg`

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. The app will open in your web browser at `http://localhost:8501`

3. Select model settings in the sidebar:
   - Model size (tiny, base, small, medium, large-v2)
   - Language (English or Spanish)
   - Compute device (CPU or CUDA for GPU acceleration)

4. Upload one or more audio files using the file uploader

5. Click "Transcribe" for each file you want to process

6. View and download the transcription results

## Project Structure

```
Transcribe_UI/
├── app.py                  # Main application file
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── utils/                  # Utility functions
│   ├── __init__.py         # Package initialization
│   ├── audio.py            # Audio processing utilities
│   ├── model.py            # Model loading and transcription
│   └── ui.py               # UI components and helpers
├── docs/                   # Documentation assets
│   └── app_screenshot.png  # Application screenshot
└── tests/                  # Unit tests
    └── test_utils.py       # Tests for utility functions
```

## Advanced Usage

### Batch Processing

For batch processing of multiple files, simply upload all files at once and click the transcribe button for each file. Results can be downloaded individually.

### Custom Language Support

While the UI currently supports English and Spanish, the underlying Whisper model supports many more languages. Advanced users can modify the language selection options in the code.

### Integration with Other Tools

The transcription output can be easily integrated with other NLP tools for further processing:
- Text summarization
- Sentiment analysis
- Translation
- Named entity recognition

## Speaker Diarization

The app includes speaker diarization capabilities to identify different speakers in your audio files.

### How Speaker Diarization Works

1. **Audio Feature Extraction**: The system extracts MFCC (Mel-frequency cepstral coefficients) features from the audio.
2. **Speaker Clustering**: Using spectral clustering algorithms, the system identifies distinct speakers based on audio characteristics.
3. **Segment Labeling**: Each segment of the transcription is labeled with a speaker identifier (e.g., SPEAKER_1, SPEAKER_2).

### Requirements for Speaker Diarization

- **FFmpeg**: Required for audio processing. If FFmpeg is not installed, the app will attempt to use PyDub as a fallback.
- **SoundFile**: Used for reliable audio loading.
- **scikit-learn**: Used for the clustering algorithms.

### Diarization Settings

- **Number of Speakers**: You can specify the expected number of speakers or let the system estimate it automatically.
- **Minimum Segment Duration**: Controls the minimum length of a speaker segment (default: 1.0 seconds).

### Limitations

- Speaker diarization works best with clear audio and distinct speakers
- Background noise can affect the accuracy of speaker identification
- Very short speaker turns may not be accurately identified
- The system assigns generic labels (SPEAKER_1, SPEAKER_2) rather than identifying actual individuals

## Model Information

This app uses the `faster-whisper` implementation of OpenAI's Whisper model, which offers:

- Improved performance over the original Whisper implementation
- Reduced memory usage
- Support for CPU and GPU acceleration
- Multiple model sizes to balance accuracy and speed

Model size comparison:
| Model Size | Parameters | Relative Speed | Memory Usage | Accuracy |
|------------|------------|----------------|--------------|----------|
| tiny       | 39M        | Very Fast      | Low          | Basic    |
| base       | 74M        | Fast           | Low          | Good     |
| small      | 244M       | Medium         | Medium       | Better   |
| medium     | 769M       | Slow           | High         | Great    |
| large-v2   | 1550M      | Very Slow      | Very High    | Best     |

## Requirements

- Python 3.7+
- FFmpeg (for audio processing and diarization)
- Streamlit
- faster-whisper
- NumPy
- PyDub
- SoundFile (for audio processing)
- scikit-learn (for speaker diarization)
- librosa (for audio feature extraction)

See `requirements.txt` for specific version requirements.

## Performance Considerations

- Larger models provide better accuracy but require more memory and processing power
- GPU acceleration (CUDA) significantly improves performance but requires a compatible NVIDIA GPU
- First-time model usage will download the model files (one-time process)
- Processing long audio files (>10 minutes) may take significant time with larger models
- Consider using the 'tiny' or 'base' model for quick testing before using larger models

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
