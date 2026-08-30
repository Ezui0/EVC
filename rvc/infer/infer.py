import asyncio
import gc
import os
import tempfile

import edge_tts
import gradio as gr
import torch
from pydub import AudioSegment
from scipy.io import wavfile

from audio_separator.separator import Separator
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





def separate_vox(input_path, output_dir=None):
    """
    Separate audio using UVR models.
    
    Args:
        input_path: Path to input audio file
        output_dir: Output directory for separated files
    
    Returns:
        Tuple of (vocals, instrumental, lead_vocals, backing_vocals, vocals_no_reverb, vocals_reverb)
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    separator = Separator(output_dir=output_dir)
    
    # Vocals and Instrumental
    vocals = os.path.join(output_dir, 'Vocals.wav')
    instrumental = os.path.join(output_dir, 'Instrumental.wav')

    # Vocals with Reverb and Vocals without Reverb
    vocals_reverb = os.path.join(output_dir, 'Vocals (Reverb).wav')
    vocals_no_reverb = os.path.join(output_dir, 'Vocals (No Reverb).wav')
    
    # Lead Vocals and Backing Vocals
    lead_vocals = os.path.join(output_dir, 'Lead Vocals.wav')
    backing_vocals = os.path.join(output_dir, 'Backing Vocals.wav')

    # Splitting a track into Vocal and Instrumental
    separator.load_model(model_filename='model_bs_roformer_ep_317_sdr_12.9755.ckpt')
    voc_inst = separator.separate(input_path)
    
    os.rename(os.path.join(output_dir, voc_inst[0]), instrumental)
    os.rename(os.path.join(output_dir, voc_inst[1]), vocals)
    
    # Applying DeEcho-DeReverb to Vocals
    separator.load_model(model_filename='UVR-DeEcho-DeReverb.pth')
    voc_no_reverb = separator.separate(vocals)
    os.rename(os.path.join(output_dir, voc_no_reverb[0]), vocals_no_reverb)
    os.rename(os.path.join(output_dir, voc_no_reverb[1]), vocals_reverb)

    # Separating Back Vocals from Main Vocals
    separator.load_model(model_filename='mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt')
    backing_voc = separator.separate(vocals_no_reverb)
    os.rename(os.path.join(output_dir, backing_voc[0]), backing_vocals)
    os.rename(os.path.join(output_dir, backing_voc[1]), lead_vocals)

    return vocals, instrumental, lead_vocals, backing_vocals, vocals_no_reverb, vocals_reverb


def merge_audio(audio1_path, audio2_path, output_path, volume_ratio=1.0):
    """
    Merge two audio files together.
    
    Args:
        audio1_path: Path to first audio file
        audio2_path: Path to second audio file
        output_path: Path for merged output
        volume_ratio: Volume ratio for audio2 relative to audio1 (default: 1.0)
    """
    audio1 = AudioSegment.from_file(audio1_path)
    audio2 = AudioSegment.from_file(audio2_path)
    
    # Ensure both audio have same length (pad or trim)
    if len(audio1) > len(audio2):
        audio2 = audio2 + AudioSegment.silent(duration=len(audio1) - len(audio2))
    elif len(audio2) > len(audio1):
        audio1 = audio1 + AudioSegment.silent(duration=len(audio2) - len(audio1))
    
    # Adjust volume of audio2 if needed
    if volume_ratio != 1.0:
        audio2 = audio2.apply_gain(volume_ratio)
    
    # Overlay audio2 on top of audio1
    merged = audio1.overlay(audio2)
    
    # Export merged audio
    merged.export(output_path, format=os.path.splitext(output_path)[1].lstrip("."))
    return output_path


def convert_with_uvr(
    rvc_model=None,
    input_path=None,
    use_uvr=True,
    is_backing=False,
    f0_method="rmvpe",
    hop_length=128,
    index_rate=0,
    f0_min=50,
    f0_max=1100,
    protect=0.5,
    volume_envelope=1,
    rvc_pitch=0,
    output_format="wav",
    backing_volume=1.0,
):
    """
    Convert audio with optional UVR separation.
    
    Args:
        rvc_model: RVC model name
        input_path: Input audio file path
        use_uvr: Whether to use UVR separation
        is_backing: If True, use backing vocals; if False, use lead vocals
        f0_method: Pitch extraction method
        hop_length: Hop length for pitch extraction
        index_rate: Index rate for RVC
        f0_min: Minimum pitch
        f0_max: Maximum pitch
        protect: Protection level
        volume_envelope: Volume envelope
        rvc_pitch: Pitch shift
        output_format: Output audio format
        backing_volume: Volume ratio for backing vocals when merged (default: 1.0)
    
    Returns:
        Path to converted audio file
    """
    if not rvc_model:
        raise gr.Error("Please select a voice model to convert.")
    if not input_path:
        raise gr.Error("Please select or upload an audio file to convert.")
    if not os.path.exists(input_path):
        raise gr.Error(f"Could not find the file '{input_path}'.")

    # If UVR is enabled, separate the audio first
    if use_uvr:
        print_display_progress(0.1, "[🎵] Separating audio with UVR...")
        vocals, instrumental, lead_vocals, backing_vocals, vocals_no_reverb, vocals_reverb = separate_vox(input_path)
        
        if is_backing:
            # Convert both lead and backing vocals separately
            print_display_progress(0.2, "[🎤] Converting lead vocals...")
            lead_converted = _run_conversion(
                rvc_model=rvc_model,
                input_path=lead_vocals,
                f0_method=f0_method,
                hop_length=hop_length,
                index_rate=index_rate,
                f0_min=f0_min,
                f0_max=f0_max,
                protect=protect,
                volume_envelope=volume_envelope,
                rvc_pitch=rvc_pitch,
                output_format="wav",  # Keep as wav for merging
            )
            
            print_display_progress(0.5, "[🎤] Converting backing vocals...")
            backing_converted = _run_conversion(
                rvc_model=rvc_model,
                input_path=backing_vocals,
                f0_method=f0_method,
                hop_length=hop_length,
                index_rate=index_rate,
                f0_min=f0_min,
                f0_max=f0_max,
                protect=protect,
                volume_envelope=volume_envelope,
                rvc_pitch=rvc_pitch,
                output_format="wav",  # Keep as wav for merging
            )
            
            # Merge lead and backing vocals
            print_display_progress(0.8, "[🔊] Merging lead and backing vocals...")
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            if len(base_name) > 50:
                base_name = "Made_in_EVC"
            
            merged_path = os.path.join(OUTPUT_DIR, f"{base_name}_merged._{rvc_model}_backing.{output_format}")
            
            # Merge with backing volume adjustment
            merge_audio(lead_converted, backing_converted, merged_path, backing_volume)
            
            # Clean up temporary files
            if os.path.exists(lead_converted) and os.path.basename(lead_converted).startswith("temp_"):
                os.remove(lead_converted)
            if os.path.exists(backing_converted) and os.path.basename(backing_converted).startswith("temp_"):
                os.remove(backing_converted)
            
            output_path = merged_path
            print_display_progress(1.0, "[✅] Conversion with merged vocals complete!")
        else:
            # Convert only lead vocals
            print_display_progress(0.2, "[🎤] Using lead vocals for conversion...")
            output_path = _run_conversion(
                rvc_model=rvc_model,
                input_path=lead_vocals,
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
    else:
        # Convert original audio without UVR
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
    
    return output_path


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
        base_name = "Made_in_PolGen"
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
def rvc_infer(
    rvc_model=None,
    input_path=None,
    use_uvr=False,
    is_backing=False,
    f0_method="rmvpe",
    hop_length=128,
    index_rate=0,
    f0_min=50,
    f0_max=1100,
    protect=0.5,
    volume_envelope=1,
    rvc_pitch=0,
    f0_file=None,
    output_format="wav",
    backing_volume=1.0,
):
    if use_uvr:
        output_path = convert_with_uvr(
            rvc_model=rvc_model,
            input_path=input_path,
            use_uvr=True,
            is_backing=is_backing,
            f0_method=f0_method,
            hop_length=hop_length,
            index_rate=index_rate,
            f0_min=f0_min,
            f0_max=f0_max,
            protect=protect,
            volume_envelope=volume_envelope,
            rvc_pitch=rvc_pitch,
            output_format=output_format,
            backing_volume=backing_volume,
        )
    else:
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
def rvc_batch_infer(
    rvc_model=None,
    input_path=None,
    use_uvr=False,
    is_backing=False,
    f0_method="rmvpe",
    hop_length=128,
    index_rate=0,
    f0_min=50,
    f0_max=1100,
    protect=0.5,
    volume_envelope=1,
    rvc_pitch=0,
    backing_volume=1.0,
    *unused,
):
    message, _audio = rvc_infer(
        rvc_model=rvc_model,
        input_path=input_path,
        use_uvr=use_uvr,
        is_backing=is_backing,
        f0_method=f0_method,
        hop_length=hop_length,
        index_rate=index_rate,
        f0_min=f0_min,
        f0_max=f0_max,
        protect=protect,
        volume_envelope=volume_envelope,
        rvc_pitch=rvc_pitch,
        backing_volume=backing_volume,
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
