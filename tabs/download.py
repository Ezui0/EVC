from rvc.modules.model_manager import download_from_url, upload_separate_files, upload_zip_file
import gradio as gr
import yt_dlp
import os, sys
from urllib.parse import urlparse, parse_qs
from contextlib import suppress

BASE_DIR = os.getcwd()

output_dir = os.path.join(BASE_DIR, 'audios')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def raise_exception(error_msg):
    raise gr.Error(error_msg)

app_name = "Polgen RVC"
url_github = "https://github.com/Bebra777228/PolGen-RVC"


def get_youtube_video_id(url, ignore_playlist=True):
    """Extract YouTube video ID from various URL formats"""
    query = urlparse(url)
    
    # Handle youtu.be URLs
    if query.hostname == 'youtu.be':
        video_id = query.path[1:]
        if video_id:
            return video_id
        return None

    # Handle youtube.com URLs
    if query.hostname in {'www.youtube.com', 'youtube.com', 'music.youtube.com'}:
        if query.path == '/watch':
            video_id = parse_qs(query.query).get('v', [None])[0]
            if video_id:
                return video_id
        elif query.path.startswith('/watch/'):
            return query.path.split('/')[2] if len(query.path.split('/')) > 2 else None
        elif query.path.startswith('/embed/'):
            return query.path.split('/')[2] if len(query.path.split('/')) > 2 else None
        elif query.path.startswith('/v/'):
            return query.path.split('/')[2] if len(query.path.split('/')) > 2 else None
        elif not ignore_playlist:
            # For playlist URLs, extract playlist ID
            playlist_id = parse_qs(query.query).get('list', [None])[0]
            if playlist_id:
                return playlist_id
    
    return None


def yt_download(link):
    """Download audio from YouTube"""
    if not link or not link.strip():
        error_msg = 'No URL provided. Please enter a valid YouTube URL.'
        raise_exception(error_msg)
    
    # Validate URL
    if urlparse(link).scheme not in ['http', 'https']:
        error_msg = 'Invalid URL. Please provide a valid YouTube URL.'
        raise_exception(error_msg)
    
    # Check if it's a valid YouTube URL
    song_id = get_youtube_video_id(link)
    if song_id is None:
        error_msg = 'Invalid YouTube URL. Please check the URL and try again.'
        raise_exception(error_msg)
    
    ydl_opts = {
        'format': 'bestaudio',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'nocheckcertificate': True,
        'ignoreerrors': True,
        'no_warnings': True,
        'quiet': True,
        'extractaudio': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Download the audio
            ydl.download([link])
            
            # Get the downloaded file info
            info = ydl.extract_info(link, download=False)
            title = info.get('title', 'audio')
            download_path = os.path.join(output_dir, f"{title}.mp3")
            
            # Check if file exists
            if not os.path.exists(download_path):
                # Try alternative naming (sometimes yt-dlp adds suffixes)
                import glob
                files = glob.glob(os.path.join(output_dir, f"{title}*.mp3"))
                if files:
                    download_path = files[0]
                else:
                    error_msg = 'Failed to download or locate the audio file.'
                    raise_exception(error_msg)
            
            return download_path
            
    except Exception as e:
        error_msg = f'Error downloading from YouTube: {str(e)}'
        raise_exception(error_msg)
        return None


def dltabs():
    with gr.TabItem("Download Music"):
        url_input = gr.Textbox(label="URL YT", placeholder="Enter YouTube URL here...")
        optau = gr.Audio(label="OPT", type="filepath")
        dl_yt = gr.Button("Download")
        dl_yt.click(fn=yt_download, inputs=[url_input], outputs=[optau])
    
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
