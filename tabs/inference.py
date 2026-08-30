import gradio as gr
from rvc.infer.infer import (
    rvc_infer,
    rvc_batch_infer,
    rvc_edgetts_infer,
    RVC_MODELS_DIR,
)
from rvc.modules.model_manager import download_from_url, upload_separate_files, upload_zip_file
from tabs.components.modules import (
    OUTPUT_FORMAT,
    edge_voices,
    update_edge_voices,
    get_folders,
    update_models_list,
)
import os

app_name = "Polgen RVC"
url_github = "https://github.com/Bebra777228/PolGen-RVC"

def get_model_names():
    if not os.path.exists(RVC_MODELS_DIR):
        return []
    return [d for d in os.listdir(RVC_MODELS_DIR) if os.path.isdir(os.path.join(RVC_MODELS_DIR, d))]

def get_indexes():
    model_names = get_model_names()
    indexes = []
    for model in model_names:
        model_dir = os.path.join(RVC_MODELS_DIR, model)
        if os.path.exists(model_dir):
            for f in os.listdir(model_dir):
                if f.endswith('.index'):
                    indexes.append(os.path.join(model_dir, f))
    return indexes

def get_index():
    indexes = get_indexes()
    return indexes[0] if indexes else ""

def change_choices():
    models = get_folders()
    indexes = get_indexes()
    return gr.Dropdown(choices=models, value=models[0] if models else ""), gr.Dropdown(choices=indexes, value=indexes[0] if indexes else "")

def save_to_wav(file):
    return file

def save_to_wav2(file):
    if file is None:
        return ""
    return file.name if hasattr(file, 'name') else str(file)

def get_audio_dropdown_update(current_value=None):
    """Builds a gr.update for the audio-selection dropdown: the list of files from the "audios"
    folder plus the currently selected path (recorded/uploaded), so the value always matches choices."""
    audio_files = []
    if os.path.exists("audios"):
        audio_files = [os.path.join("audios", f) for f in os.listdir("audios") if f.endswith(('.wav', '.mp3', '.flac'))]
    choices = list(dict.fromkeys([""] + ([current_value] if current_value else []) + audio_files))
    value = current_value if current_value else (audio_files[0] if audio_files else "")
    return gr.update(choices=choices, value=value)

def change_choices2():
    if os.path.exists("audios"):
        audio_files = [os.path.join("audios", f) for f in os.listdir("audios") if f.endswith(('.wav', '.mp3', '.flac'))]
        return gr.Dropdown(choices=[""] + audio_files, value=audio_files[0] if audio_files else "")
    return gr.Dropdown(choices=[""], value="")

def inference_tab():
    # Folder for audio files that the user adds via dropbox/recording
    os.makedirs("audios", exist_ok=True)
    with gr.Tabs():
        with gr.TabItem("Inference"):
            gr.HTML(f"<h1> Easy GUI v2 (rejekts) - adapted to {app_name} 💻 </h1>")

            with gr.Row():
                sid0 = gr.Dropdown(label="1. Choose your Model.", choices=get_folders(), value=get_folders()[0] if get_folders() else None)
                refresh_button = gr.Button("Refresh", variant="primary")
                vc_transform0 = gr.Number(label="Optional: Change pitch here or leave at 0.", value=0)
                spk_item = gr.Slider(minimum=0,maximum=2333,step=1,label="Please select speaker id",value=0,visible=False,interactive=True)
                but0 = gr.Button("Convert", variant="primary")
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        dropbox = gr.Audio(label="Drop your audio here & hit the Reload button.")
                    with gr.Row():
                        record_button=gr.Audio(sources="microphone", label="OR Record audio.", type="filepath")
                    with gr.Row():
                        input_audio0 = gr.Dropdown(
                            label="2.Choose your audio.",
                            value="",
                            choices=[""]
                            )
                        
                        dropbox.upload(fn=lambda file: get_audio_dropdown_update(save_to_wav2(file)), 
                                     inputs=[dropbox], outputs=[input_audio0])
                        
                        refresh_button2 = gr.Button("Refresh", variant="primary", size='sm')
                        refresh_button2.click(fn=lambda: get_audio_dropdown_update(), 
                                            inputs=[], outputs=[input_audio0])
                        
                        record_button.change(fn=lambda file: get_audio_dropdown_update(save_to_wav2(file)),
                                           inputs=[record_button], outputs=[input_audio0])
                with gr.Column():
                    with gr.Accordion("Index Settings", open=True):
                        file_index1 = gr.Dropdown(
                            label="3. Path to your added.index file (if it didn't automatically find it.)",
                            choices=get_indexes(),
                            value=get_indexes()[0] if get_indexes() else None,
                            interactive=True,
                            )
                        refresh_button.click(
                            fn=lambda: (
                                gr.update(choices=get_folders(), value=get_folders()[0] if get_folders() else None),
                                gr.update(choices=get_indexes(), value=get_indexes()[0] if get_indexes() else None),
                            ),
                            inputs=[], outputs=[sid0, file_index1]
                            )
                        index_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="Search feature ratio",
                            value=0.66,
                            interactive=True,
                            )
                    vc_output2 = gr.Audio(label="Output Audio (Click on the Three Dots in the Right Corner to Download)")
                    
                    # NEW SETTINGS REPLACING OLD ONES
                    with gr.Accordion("Conversion Settings", open=False):
                        with gr.Column(variant="panel"):
                            with gr.Accordion("Standard Settings", open=False):
                                with gr.Group():
                                    with gr.Column():
                                        f0_method = gr.Dropdown(
                                            value="rmvpe",
                                            label="Pitch extraction method",
                                            choices=["rmvpe", "fcpe", "crepe", "crepe-tiny"],
                                            interactive=True,
                                            visible=True,
                                        )
                                        hop_length = gr.Slider(
                                            minimum=8,
                                            maximum=512,
                                            step=8,
                                            value=128,
                                            label="Hop length",
                                            info="Smaller values lead to longer conversions, increasing risk of voice artifacts, but achieve more accurate pitch transfer.",
                                            interactive=True,
                                            visible=False,
                                        )
                                        index_rate = gr.Slider(
                                            minimum=0,
                                            maximum=1,
                                            step=0.1,
                                            value=0,
                                            label="Index influence",
                                            info="Influence of the index file; Higher values mean more influence. Lower values can help soften artifacts in the audio.",
                                            interactive=True,
                                            visible=True,
                                        )
                                        volume_envelope = gr.Slider(
                                            minimum=0,
                                            maximum=1,
                                            step=0.01,
                                            value=1,
                                            label="RMS mix rate",
                                            info="Replace or mix with the output volume envelope. Closer to 1 uses more output envelope.",
                                            interactive=True,
                                            visible=True,
                                        )
                                        protect = gr.Slider(
                                            minimum=0,
                                            maximum=0.5,
                                            step=0.01,
                                            value=0.5,
                                            label="Consonant protection",
                                            info="Protect consonants and breath sounds to avoid electroacoustic breaks. Max value 0.5 provides full protection.",
                                            interactive=True,
                                            visible=True,
                                        )

                            with gr.Accordion("Advanced Settings", open=False):
                                with gr.Column():
                                    with gr.Row():
                                        f0_min = gr.Slider(minimum=1,maximum=120,step=1,value=50,label="Minimum pitch range",info="Defines the lower bound of the pitch range for fundamental frequency (F0) detection.",interactive=True,visible=True)                                                                                                                                                                                                                                                                                                               
                                        f0_max = gr.Slider(minimum=380,maximum=16000,step=1,value=1100,label="Maximum pitch range",info="Defines the upper bound of the pitch range for fundamental frequency (F0) detection.",interactive=True,visible=True)
                                    with gr.Row():
                                        use_uvr = gr.Checkbox(label="Use UVR for Separating Vocals", value=False)
                                        is_backing = gr.Checkbox(label="Use backing vocal", value=False)
                                        backing_volume = gr.Slider(
                                            minimum=0.0,
                                            maximum=2.0,
                                            step=0.1,
                                            value=1.0,
                                            label="Backing vocal volume",
                                            info="Adjust the volume of backing vocals relative to lead vocals",
                                            interactive=True,
                                            visible=True,
                                        )
                                        output_format = gr.Dropdown(
                                            label="Output format",
                                            choices=["wav", "mp3", "flac", "ogg", "m4a"],
                                            value="wav",
                                            interactive=True,
                                        )
                    
            with gr.Row():
                vc_output1 = gr.Textbox("")
                f0_file = gr.File(label="F0 curve file (optional)", visible=False)
                
                but0.click(
                    rvc_infer,
                    [
                        sid0,
                        input_audio0,
                        use_uvr,  # Pass the checkbox value
                        is_backing,  # Pass the checkbox value
                        f0_method,
                        hop_length,  
                        index_rate,
                        f0_min,      
                        f0_max,
                        protect,     
                        volume_envelope, 
                        vc_transform0,
                        f0_file,
                        output_format,
                        backing_volume
                    ],
                    [vc_output1, vc_output2],
                )
            with gr.Accordion("Batch Conversion",open=False, visible=False):
                with gr.Row():
                    with gr.Column():
                        vc_transform1 = gr.Number(
                            label="Transpose (semitones)", value=0
                        )
                        opt_input = gr.Textbox(label="Output folder", value="opt")
                        f0method1 = gr.Radio(
                            label="Pitch extraction algorithm",
                            choices=["pm", "harvest", "crepe"],
                            value="pm",
                            interactive=True,
                        )
                        filter_radius1 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label="Median filtering radius",
                            value=3,
                            step=1,
                            interactive=True,
                        )
                    with gr.Column():
                        file_index3 = gr.Textbox(
                            label="Feature index path",
                            value="",
                            interactive=True,
                        )
                        file_index4 = gr.Dropdown(
                            label="Auto detect index path",
                            choices=get_indexes(),
                            interactive=True,
                        )
                        refresh_button.click(
                            fn=lambda: gr.update(choices=get_indexes()),
                            inputs=[],
                            outputs=file_index4,
                        )
                        index_rate2 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="Search feature ratio",
                            value=1,
                            interactive=True,
                        )
                    with gr.Column():
                        resample_sr1 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label="Resample to final sample rate",
                            value=0,
                            step=1,
                            interactive=True,
                        )
                        rms_mix_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="Volume envelope mix ratio",
                            value=1,
                            interactive=True,
                        )
                        protect1 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label="Protect voiceless consonants",
                            value=0.33,
                            step=0.01,
                            interactive=True,
                        )
                    with gr.Column():
                        dir_input = gr.Textbox(
                            label="Input folder path",
                            value="",
                        )
                        inputs = gr.File(
                            file_count="multiple", label="Or batch input audio files"
                        )
                    with gr.Row():
                        format1 = gr.Radio(
                            label="Export format",
                            choices=OUTPUT_FORMAT,
                            value="flac",
                            interactive=True,
                        )
                        but1 = gr.Button("Convert", variant="primary")
                        vc_output3 = gr.Textbox(label="Output info")
                    
                    but1.click(
                        rvc_batch_infer,
                        [
                            sid0,
                            dir_input,
                            use_uvr,  # Pass the checkbox value
                            is_backing,  # Pass the checkbox value
                            f0_method,
                            hop_length,
                            index_rate,
                            f0_min,
                            f0_max,
                            protect,
                            volume_envelope,
                            vc_transform1,
                            backing_volume
                        ],
                        [vc_output3],
                    )

        with gr.TabItem("Text to Speech + RVC"):
            default_language = list(edge_voices.keys())[0]
            with gr.Row():
                tts_language = gr.Dropdown(
                    label="Select Language",
                    choices=list(edge_voices.keys()),
                    value=default_language
                )
                tts_voice = gr.Dropdown(
                    label="Select TTS Voice",
                    choices=edge_voices[default_language],
                    value=edge_voices[default_language][0]
                )
                tts_language.change(
                    fn=update_edge_voices,
                    inputs=[tts_language],
                    outputs=[tts_voice]
                )
            with gr.Row():
                tts_text = gr.Textbox(label="Enter text to synthesize", lines=5)
            with gr.Row():
                tts_rate = gr.Slider(minimum=-100, maximum=100, value=0, label="Rate (%)")
                tts_volume = gr.Slider(minimum=-100, maximum=100, value=0, label="Volume (%)")
                tts_pitch = gr.Slider(minimum=-100, maximum=100, value=0, label="Pitch (Hz)")
            with gr.Row():
                tts_output = gr.Audio(label="TTS Output")
            with gr.Row():
                tts_rvc_model = gr.Dropdown(label="RVC Model for conversion", choices=get_folders())
                refresh_models_btn = gr.Button("Refresh Models", variant="secondary", size="sm")
                refresh_models_btn.click(
                    fn=update_models_list,
                    inputs=[],
                    outputs=[tts_rvc_model]
                )
            with gr.Row():
                tts_convert_btn = gr.Button("Convert TTS to RVC", variant="primary")
            
            tts_convert_btn.click(
                rvc_edgetts_infer,
                inputs=[
                    tts_rvc_model,
                    f0_method,
                    hop_length,
                    index_rate,
                    f0_min,
                    f0_max,
                    protect,
                    volume_envelope,
                    vc_transform0,
                    tts_voice,
                    tts_text,
                    tts_rate,
                    tts_volume,
                    tts_pitch,
                    output_format
                ],
                outputs=tts_output
            )

        with gr.TabItem("Download Model"):
            output_message = gr.Textbox(label="Output Message", interactive=False)
            
            with gr.Accordion("Download ZIP from URL", open=True):
                gr.HTML(
                    "<h3>"
                    "Supported sites: "
                    "<a href='https://huggingface.co/' target='_blank'>HuggingFace</a>, "
                    "<a href='https://pixeldrain.com/' target='_blank'>Pixeldrain</a>, "
                    "<a href='https://drive.google.com/' target='_blank'>Google Drive</a>, "
                    "<a href='https://mega.nz/' target='_blank'>Mega</a>, "
                    "<a href='https://disk.yandex.ru/' target='_blank'>Yandex Disk</a>"
                    "</h3>"
                )
                with gr.Column():
                    with gr.Group():
                        zip_link = gr.Textbox(label="ZIP download link")
                        model_name = gr.Textbox(
                            label="Model name",
                            info="Give your uploaded model a unique name different from other voice models.",
                        )
                    download_btn = gr.Button("Download model", variant="primary")
                
                download_btn.click(
                    download_from_url,
                    inputs=[zip_link, model_name],
                    outputs=output_message,
                )
            
            with gr.Accordion("Upload ZIP file", open=False):
                with gr.Column():
                    with gr.Group():
                        zip_file = gr.File(label="Zip file", file_types=[".zip"], file_count="single")
                        model_name_zip = gr.Textbox(
                            label="Model name",
                            info="Give your uploaded model a unique name different from other voice models.",
                        )
                    upload_zip_btn = gr.Button("Upload model", variant="primary")
                
                upload_zip_btn.click(
                    upload_zip_file,
                    inputs=[zip_file, model_name_zip],
                    outputs=output_message,
                )
            
            with gr.Accordion("Upload .pth and .index files", open=False):
                with gr.Column():
                    with gr.Group():
                        with gr.Row(equal_height=False):
                            pth_file = gr.File(label="pth file", file_types=[".pth"], file_count="single")
                            index_file = gr.File(label="index file", file_types=[".index"], file_count="single")
                        model_name_files = gr.Textbox(
                            label="Model name",
                            info="Give your uploaded model a unique name different from other voice models.",
                        )
                    upload_files_btn = gr.Button("Upload model", variant="primary")
                
                upload_files_btn.click(
                    upload_separate_files,
                    inputs=[pth_file, index_file, model_name_files],
                    outputs=output_message,
                )
            
            with gr.Row():
                gr.Markdown(
                f"""
                Original RVC: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
                {app_name}: {url_github}
                """
                )
