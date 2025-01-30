from flask import Flask, request, render_template, send_file, url_for, Response, stream_with_context
import os
import torchaudio
from werkzeug.utils import secure_filename
from demucs.pretrained import get_model
from demucs.audio import AudioFile
from demucs.apply import apply_model
import subprocess
import time

app = Flask(__name__)

# Configure upload and output folders
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "separated_audio"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure upload and output folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Global variable to store progress
current_progress = 0  # Renamed to avoid conflict


def separate_drums(input_file, output_dir):
    """
    Separates the drum audio from a given input song using Demucs.

    Args:
        input_file (str): Path to the input audio file.
        output_dir (str): Directory to save the separated audio.

    Returns:
        str: Path to the separated drum audio file.
    """
    global current_progress

    # Load the pre-trained Demucs model
    model = get_model(name="htdemucs")
    model.cpu()  # Use .cuda() if a GPU is available

    # Load the input audio file using Demucs' AudioFile class
    wav = AudioFile(input_file).read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)

    # Separate the audio into its sources (drums, vocals, etc.)
    def update_progress(p):
        global current_progress
        current_progress = int(p * 100)  # Convert progress to percentage

    sources = apply_model(model, wav[None], split=True, overlap=0.25, progress=update_progress)[0]
    sources = sources.cpu()

    # Generate path for the separated drum audio
    track_name = os.path.splitext(os.path.basename(input_file))[0]
    drum_path = os.path.join(output_dir, f"{track_name}_drums.wav")

    # Save the separated drum audio
    torchaudio.save(drum_path, sources[0], model.samplerate, format="wav")
    return drum_path


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Convert to WAV if necessary
        if filename.lower().endswith('.mp3'):
            wav_path = os.path.splitext(file_path)[0] + '.wav'
            # Overwrite the WAV file if it already exists
            if os.path.exists(wav_path):
                os.remove(wav_path)
            command = ['ffmpeg', '-i', file_path, '-ac', '1', wav_path]
            subprocess.run(command)
            input_path = wav_path
        else:
            input_path = file_path

        # Separate the drums
        output_file = separate_drums(input_path, app.config['OUTPUT_FOLDER'])

        # Generate the download URL using url_for
        download_url = url_for('download_file', filename=os.path.basename(output_file))

        # Send the file as a downloadable response
        return render_template('download.html', download_url=download_url)


@app.route('/progress')
def progress():
    def generate():
        global current_progress
        while current_progress < 100:
            yield f"data: {current_progress}\n\n"
            time.sleep(0.5)  # Send updates every 0.5 seconds
        yield f"data: 100\n\n"  # Ensure 100% is sent
    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@app.route('/download/<path:filename>', methods=['GET'])
def download_file(filename):
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    return send_file(file_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)