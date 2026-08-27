import gradio as gr
from rvc.infer.infer import (
    rvc_infer,
    rvc_edgetts_infer,
    RVC_MODELS_DIR,
    OUTPUT_DIR,
    load_rvc_model,
    get_vc
)
from rvc.modules.model_manager import download_from_url, upload_separate_files, upload_zip_file
from tabs.components.modules import (
    OUTPUT_FORMAT,
    edge_voices,
    update_edge_voices,
    get_folders,
    update_models_list,
    process_file_upload,
    show_hop_slider,
    swap_visibility,
    swap_buttons
)
import os
import json

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

def change_choices2():
    if os.path.exists("audios"):
        audio_files = [os.path.join("audios", f) for f in os.listdir("audios") if f.endswith(('.wav', '.mp3', '.flac'))]
        return gr.Dropdown(choices=[""] + audio_files, value=audio_files[0] if audio_files else "")
    return gr.Dropdown(choices=[""], value="")

# Fix for the inference function to handle None values
def safe_rvc_infer(*args):
    # Filter out None values and provide defaults
    cleaned_args = []
    for arg in args:
        if arg is None:
            cleaned_args.append("")  # or appropriate default
        else:
            cleaned_args.append(arg)
    
    # Ensure we have the right number of arguments
    while len(cleaned_args) < 11:
        cleaned_args.append("")  # default empty value
    
    try:
        result = rvc_infer(*cleaned_args[:11])
        return result
    except Exception as e:
        print(f"Error in inference: {e}")
        return "Error occurred", None

def safe_rvc_edgetts_infer(*args):
    # Filter out None values
    cleaned_args = []
    for arg in args:
        if arg is None:
            cleaned_args.append("")  # or appropriate default
        else:
            cleaned_args.append(arg)
    
    while len(cleaned_args) < 15:
        cleaned_args.append("")
    
    try:
        result = rvc_edgetts_infer(*cleaned_args[:15])
        return result
    except Exception as e:
        print(f"Error in TTS inference: {e}")
        return None

def safe_download_from_url(zip_link, model_name):
    if zip_link is None or zip_link == "":
        return "Error: Please provide a valid ZIP download link"
    if model_name is None or model_name == "":
        return "Error: Please provide a model name"
    try:
        return download_from_url(zip_link, model_name)
    except Exception as e:
        return f"Error downloading model: {str(e)}"

def safe_upload_zip_file(zip_file, model_name):
    if zip_file is None:
        return "Error: Please upload a ZIP file"
    if model_name is None or model_name == "":
        return "Error: Please provide a model name"
    try:
        return upload_zip_file(zip_file, model_name)
    except Exception as e:
        return f"Error uploading ZIP: {str(e)}"

def safe_upload_separate_files(pth_file, index_file, model_name):
    if pth_file is None:
        return "Error: Please upload a .pth file"
    if model_name is None or model_name == "":
        return "Error: Please provide a model name"
    try:
        return upload_separate_files(pth_file, index_file, model_name)
    except Exception as e:
        return f"Error uploading files: {str(e)}"

def inference_tab():
    with gr.Tabs():
        with gr.TabItem("Inference"):
            gr.HTML(f"<h1> Easy GUI v2 (rejekts) - adapted to {app_name} 💻 </h1>")

            with gr.Row():
                sid0 = gr.Dropdown(label="1.Choose your Model.", choices=get_folders(), value=get_folders()[0] if get_folders() else '')
                refresh_button = gr.Button("Refresh", variant="primary")
                vc_transform0 = gr.Number(label="Optional: You can change the pitch here or leave it at 0.", value=0)
                spk_item = gr.Slider(
                    minimum=0,
                    maximum=2333,
                    step=1,
                    label="Please select speaker id",
                    value=0,
                    visible=False,
                    interactive=True,
                )
                but0 = gr.Button("Convert", variant="primary")
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        dropbox = gr.File(label="Drop your audio here & hit the Reload button.")
                    with gr.Row():
                        record_button=gr.Audio(sources="microphone", label="OR Record audio.", type="filepath")
                    with gr.Row():
                        input_audio0 = gr.Dropdown(
                            label="2.Choose your audio.",
                            value="",
                            choices=[]
                            )
                        # Fix: Use event handlers properly
                        def update_audio_from_dropbox(file):
                            val = save_to_wav2(file)
                            return gr.Dropdown(update=True, value=val, choices=[val] + (change_choices2().choices if hasattr(change_choices2(), 'choices') else []))
                        
                        dropbox.upload(fn=lambda file: (save_to_wav2(file), gr.Dropdown(update=True, choices=change_choices2().choices, value=save_to_wav2(file))), 
                                     inputs=[dropbox], outputs=[input_audio0, input_audio0])
                        
                        refresh_button2 = gr.Button("Refresh", variant="primary", size='sm')
                        refresh_button2.click(fn=lambda: gr.Dropdown(update=True, choices=change_choices2().choices), 
                                            inputs=[], outputs=[input_audio0])
                        
                        def update_from_recording(file):
                            val = save_to_wav(file)
                            return gr.Dropdown(update=True, value=val, choices=[val] + (change_choices2().choices if hasattr(change_choices2(), 'choices') else []))
                        
                        record_button.change(fn=lambda file: (save_to_wav(file), gr.Dropdown(update=True, choices=change_choices2().choices, value=save_to_wav(file))),
                                           inputs=[record_button], outputs=[input_audio0, input_audio0])
                with gr.Column():
                    with gr.Accordion("Index Settings", open=True):
                        file_index1 = gr.Dropdown(
                            label="3. Path to your added.index file (if it didn't automatically find it.)",
                            choices=get_indexes(),
                            value=get_index(),
                            interactive=True,
                            )
                        refresh_button.click(
                            fn=lambda: (gr.Dropdown(update=True, choices=get_folders(), value=get_folders()[0] if get_folders() else ""), 
                                      gr.Dropdown(update=True, choices=get_indexes(), value=get_index())),
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
                    f0method0 = gr.Radio(
                            label="Optional: Change the Pitch Extraction Algorithm.",
                            choices=["pm", "harvest", "dio", "crepe", "crepe-tiny", "mangio-crepe", "mangio-crepe-tiny"],
                            value="pm",
                            interactive=True,
                        )
                    with gr.Accordion("More", open=False):
                        crepe_hop_length = gr.Slider(
                            minimum=1,
                            maximum=512,
                            step=1,
                            label="crepe_hop_length",
                            value=160,
                            interactive=True
                            )
                        filter_radius0 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label="Median filtering radius (>=3 enables)",
                            value=3,
                            step=1,
                            interactive=True,
                            )
                        resample_sr0 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label="Resample to final sample rate (0 = no resample)",
                            value=0,
                            step=1,
                            interactive=True,
                            )
                        rms_mix_rate0 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="Volume envelope mix ratio",
                            value=1,
                            interactive=True,
                            )
                        protect0 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label="Protect voiceless consonants",
                            value=0.33,
                            step=0.01,
                            interactive=True,
                            )
            with gr.Row():
                vc_output1 = gr.Textbox("")
                f0_file = gr.File(label="F0 curve file (optional)", visible=False)
                
                # Fix: Use safe wrapper for inference
                but0.click(
                    safe_rvc_infer,
                    [
                        sid0,
                        input_audio0,
                        f0method0,
                        filter_radius0,
                        index_rate1,
                        resample_sr0,
                        rms_mix_rate0,
                        protect0,
                        crepe_hop_length,
                        vc_transform0,
                        f0_file
                    ],
                    [vc_output1, vc_output2],
                )
            with gr.Accordion("Batch Conversion",open=False):
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
                            fn=lambda: gr.Dropdown(update=True, choices=get_indexes()),
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
                    # Fix: Use safe wrapper for batch inference
                    but1.click(
                        safe_rvc_infer,
                        [
                            sid0,
                            dir_input,
                            f0method1,
                            filter_radius1,
                            index_rate2,
                            resample_sr1,
                            rms_mix_rate1,
                            protect1,
                            crepe_hop_length,
                            vc_transform1,
                            None
                        ],
                        [vc_output3],
                    )

        with gr.TabItem("Text to Speech + RVC"):
            with gr.Row():
                tts_language = gr.Dropdown(
                    label="Select Language",
                    choices=list(edge_voices.keys()),
                    value="Русский"
                )
                tts_voice = gr.Dropdown(
                    label="Select TTS Voice",
                    choices=edge_voices["Русский"],
                    value=edge_voices["Русский"][0]
                )
                # Update voice dropdown when language changes
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
            # Fix: Use safe wrapper for TTS inference
            tts_convert_btn.click(
                safe_rvc_edgetts_infer,
                inputs=[
                    tts_rvc_model,
                    f0method0,
                    filter_radius0,
                    index_rate1,
                    resample_sr0,
                    rms_mix_rate0,
                    protect0,
                    crepe_hop_length,
                    vc_transform0,
                    tts_voice,
                    tts_text,
                    tts_rate,
                    tts_volume,
                    tts_pitch
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
                
                # Fix: Use safe wrapper for download
                download_btn.click(
                    safe_download_from_url,
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
                
                # Fix: Use safe wrapper for ZIP upload
                upload_zip_btn.click(
                    safe_upload_zip_file,
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
                
                # Fix: Use safe wrapper for separate files upload
                upload_files_btn.click(
                    safe_upload_separate_files,
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

if __name__ == "__main__":
    app = gr.Blocks()
    with app:
        inference_tab()
    app.queue(concurrency_count=511, max_size=1022).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
