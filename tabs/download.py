from rvc.modules.model_manager import download_from_url, upload_separate_files, upload_zip_file
import gradio as gr

app_name = "Polgen RVC"
url_github = "https://github.com/Bebra777228/PolGen-RVC"


def dltabs():
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
