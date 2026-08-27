import asyncio
import gc
import os
import tempfile

import edge_tts
import gradio as gr
import torch
from pydub import AudioSegment
from scipy.io import wavfile

from rvc.infer.config import Config
from rvc.infer.pipeline import VC
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.lib.my_utils import load_audio
from rvc.modules import fairseq

# Define folder and file paths (constants)
RVC_MODELS_DIR = os.path.join(os.getcwd(), "models", "RVC_models")
OUTPUT_DIR = os.path.join(os.getcwd(), "output", "RVC_output")
HUBERT_BASE_PATH = os.path.join(os.getcwd(), "rvc", "models", "embedders", "hubert_base.pt")

# Create folders if they do not exist
os.makedirs(RVC_MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize configuration
config = Config()

# Cache the loaded Hubert model so it is not reloaded on every conversion
_hubert_model = None


# Displays task progress.
def display_progress(percent, message, progress=gr.Progress()):
    progress(percent, desc=message)


def print_display_progress(percent, message, progress=gr.Progress()):
    print(message)
    progress(percent, desc=message)


# Loads the RVC model and index by model name.
def load_rvc_model(rvc_model):
    # Build the path to the model directory
    model_dir = os.path.join(RVC_MODELS_DIR, rvc_model)
    if not os.path.isdir(model_dir):
        raise gr.Error(
            f"\033[91mERROR!\033[0m Model {rvc_model} not found. You may have mistyped the name or used an invalid link when installing."
        )

    # List the files in the model directory
    model_files = os.listdir(model_dir)

    # Find the model file with the .pth extension
    rvc_model_path = next((os.path.join(model_dir, f) for f in model_files if f.endswith(".pth")), None)
    # Find the index file with the .index extension
    rvc_index_path = next((os.path.join(model_dir, f) for f in model_files if f.endswith(".index")), None)

    # Make sure the model file exists
    if not rvc_model_path:
        raise gr.Error(
            f"\033[91mERROR!\033[0m Model {rvc_model} not found. You may have mistyped the name or used an invalid link when installing."
        )

    return rvc_model_path, rvc_index_path


# Loads the Hubert model (cached across requests)
def load_hubert(model_path):
    global _hubert_model

    if _hubert_model is None:
        hubert = fairseq.load_model(model_path)
        hubert = hubert.to(config.device).float()
        hubert.eval()
        _hubert_model = hubert

    return _hubert_model


# Sets up the voice converter
def get_vc(model_path):
    # Load the model state from disk (fallback for non-standard checkpoints)
    try:
        cpt = torch.load(model_path, map_location="cpu", weights_only=True)
    except Exception:
        cpt = torch.load(model_path, map_location="cpu", weights_only=False)

    # Validate the model format
    if "config" not in cpt or "weight" not in cpt:
        raise gr.Error(f"Invalid format for {model_path}. Please use a voice model trained on RVC v2.")

    # Extract model parameters
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    pitch_guidance = bool(cpt.get("f0", 1))
    version = cpt.get("version", "v1")

    # vocoder = cpt.get("vocoder", "HiFi-GAN") - reserved for future use
    input_dim = 768 if version == "v2" else 256

    # Initialize the synthesizer
    net_g = Synthesizer(*cpt["config"], use_f0=pitch_guidance, input_dim=input_dim)

    # Remove the unneeded layer
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g = net_g.to(config.device).float()
    net_g.eval()

    # Initialize the voice converter object
    vc = VC(tgt_sr, config)
    return cpt, version, net_g, tgt_sr, vc


# Convert audio to stereo and the user-selected format
def convert_audio(input_audio, output_audio, output_format):
    # Load the audio file
    audio = AudioSegment.from_file(input_audio)

    # If the audio is mono, convert it to stereo
    if audio.channels == 1:
        audio = audio.set_channels(2)

    # Default format comes from the output file extension
    export_format = output_format or os.path.splitext(output_audio)[1].lstrip(".")
    export_format = export_format.lower()

    # Export the audio file in the chosen format
    audio.export(output_audio, format=export_format)


# Synthesizes text into speech using edge_tts.
async def text_to_speech(voice, text, rate, volume, pitch, output_path):
    if not -100 <= rate <= 100:
        raise ValueError("Rate must be between -100% and +100%")
    if not -100 <= volume <= 100:
        raise ValueError("Volume must be between -100% and +100%")
    if not -100 <= pitch <= 100:
        raise ValueError("Pitch must be between -100Hz and +100Hz")

    rate = f"+{rate}%" if rate >= 0 else f"{rate}%"
    volume = f"+{volume}%" if volume >= 0 else f"{volume}%"
    pitch = f"+{pitch}Hz" if pitch >= 0 else f"{pitch}Hz"

    communicate = edge_tts.Communicate(voice=voice, text=text, rate=rate, volume=volume, pitch=pitch)
    await communicate.save(output_path)


# Core conversion pipeline: returns the path to the finished file.
# Parameter order matches the inputs order of gr.Button.click in tabs/inference.py.
def _run_conversion(
    rvc_model=None,
    input_path=None,
    f0_method="rmvpe",
    hop_length=128,
    index_rate=0,
    f0_min=50,
    f0_max=1100,
    protect=0.5,
    volume_envelope=1,
    rvc_pitch=0,
    output_format="wav",
):
    if not rvc_model:
        raise gr.Error("Please select a voice model to convert.")
    if not input_path:
        raise gr.Error("Please select or upload an audio file to convert.")
    if not os.path.exists(input_path):
        raise gr.Error(
            f"Could not find the file '{input_path}'. Make sure it finished uploading or verify the path to it."
        )

    output_format = (str(output_format) or "wav").lower().lstrip(".")

    print_display_progress(0, "\n[⚙️] Starting generation pipeline...")

    # Load the Hubert model
    display_progress(0.1, "Loading Hubert model...")
    hubert_model = load_hubert(HUBERT_BASE_PATH)
    # Load the RVC model and index
    display_progress(0.2, "Loading RVC model and index...")
    model_path, index_path = load_rvc_model(rvc_model)
    # Set up the voice converter
    display_progress(0.3, "Setting up voice converter...")
    cpt, version, net_g, tgt_sr, vc = get_vc(model_path)
    pitch_guidance = bool(cpt.get("f0", 1))

    # Build the output file name
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    if len(base_name) > 50:
        gr.Warning("The file name exceeds 50 characters and will be shortened for convenience.")
        base_name = "Made_in_PolGen"  # Rename if the original name exceeds 50 characters
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}_({rvc_model}).{output_format}")

    # Load the audio file
    display_progress(0.4, "Loading audio file...")
    audio = load_audio(input_path, 16000)

    print_display_progress(0.5, f"[🌌] Converting audio - {base_name}...")
    audio_opt = vc.pipeline(
        hubert_model,
        net_g,
        0,
        audio,
        rvc_pitch,
        f0_method,
        index_path,
        index_rate,
        pitch_guidance,
        volume_envelope,
        version,
        protect,
        hop_length,
        f0_min=f0_min,
        f0_max=f0_max,
    )

    # Save the result to a temporary wav file, then export it
    # to the user-selected format with the correct extension.
    display_progress(0.6, "Saving result...")
    tmp_fd, tmp_wav_path = tempfile.mkstemp(prefix="polgen_", suffix=".wav", dir=OUTPUT_DIR)
    os.close(tmp_fd)
    try:
        wavfile.write(tmp_wav_path, tgt_sr, audio_opt)

        # Convert the file to stereo and the user-selected format
        print_display_progress(0.8, "[💫] Converting audio to stereo...")
        convert_audio(tmp_wav_path, output_path, output_format)
    finally:
        if os.path.exists(tmp_wav_path):
            os.remove(tmp_wav_path)

    # Free up memory
    display_progress(0.9, "Freeing memory...")
    del cpt, net_g, vc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print_display_progress(1.0, f"[✅] Conversion complete - {output_path}")
    return output_path


# Convert a single audio file through the GUI.
# Returns (message, audio component) under outputs=[vc_output1, vc_output2].
def rvc_infer(
    rvc_model=None,
    input_path=None,
    f0_method="rmvpe",
    hop_length=128,
    index_rate=0,
    f0_min=50,
    f0_max=1100,
    protect=0.5,
    volume_envelope=1,
    rvc_pitch=0,
    f0_file=None,  # Reserved: F0 curve file is not supported yet
    output_format="wav",
):
    output_path = _run_conversion(
        rvc_model=rvc_model,
        input_path=input_path,
        f0_method=f0_method,
        hop_length=hop_length,
        index_rate=index_rate,
        f0_min=f0_min,
        f0_max=f0_max,
        protect=protect,
        volume_envelope=volume_envelope,
        rvc_pitch=rvc_pitch,
        output_format=output_format,
    )
    message = f"[✅] Conversion complete - {os.path.basename(output_path)}"
    return message, gr.Audio(output_path, label=os.path.basename(output_path))


# Batch conversion of a folder/files; returns only a text message
# under outputs=[vc_output3].
def rvc_batch_infer(
    rvc_model=None,
    input_path=None,
    f0_method="rmvpe",
    hop_length=128,
    index_rate=0,
    f0_min=50,
    f0_max=1100,
    protect=0.5,
    volume_envelope=1,
    rvc_pitch=0,
    *unused,
):
    message, _audio = rvc_infer(
        rvc_model=rvc_model,
        input_path=input_path,
        f0_method=f0_method,
        hop_length=hop_length,
        index_rate=index_rate,
        f0_min=f0_min,
        f0_max=f0_max,
        protect=protect,
        volume_envelope=volume_envelope,
        rvc_pitch=rvc_pitch,
    )
    return message


def rvc_edgetts_infer(
    # RVC
    rvc_model=None,
    f0_method="rmvpe",
    hop_length=128,
    index_rate=0,
    f0_min=50,
    f0_max=1100,
    protect=0.5,
    volume_envelope=1,
    rvc_pitch=0,
    # EdgeTTS
    tts_voice=None,
    tts_text=None,
    tts_rate=0,
    tts_volume=0,
    tts_pitch=0,
    output_format="wav",
):
    if not tts_text:
        raise gr.Error("Please enter some text in the input field.")
    if not tts_voice:
        raise gr.Error("Please select a language and a voice for speech synthesis.")

    display_progress(0.2, "[🎙️] Synthesizing speech...")
    input_path = os.path.join(OUTPUT_DIR, "TTS_Voice.wav")
    asyncio.run(text_to_speech(tts_voice, tts_text, tts_rate, tts_volume, tts_pitch, input_path))

    output_path = _run_conversion(
        rvc_model=rvc_model,
        input_path=input_path,
        f0_method=f0_method,
        hop_length=hop_length,
        index_rate=index_rate,
        f0_min=f0_min,
        f0_max=f0_max,
        protect=protect,
        volume_envelope=volume_envelope,
        rvc_pitch=rvc_pitch,
        output_format=output_format,
    )

    return gr.Audio(output_path, label=os.path.basename(output_path))
