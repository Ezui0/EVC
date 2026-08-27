import os
import re

import gradio as gr

from rvc.infer.infer import RVC_MODELS_DIR

OUTPUT_FORMAT = ["wav", "flac", "mp3", "ogg", "opus", "m4a", "aiff", "ac3"]

edge_voices = {
    "English (UK)": ["en-GB-SoniaNeural", "en-GB-RyanNeural"],
    "English (US)": ["en-US-JennyNeural", "en-US-GuyNeural"],
    "Arabic (Egypt)": ["ar-EG-SalmaNeural", "ar-EG-ShakirNeural"],
    "Arabic (Saudi Arabia)": ["ar-SA-HamedNeural", "ar-SA-ZariyahNeural"],
    "Bengali (Bangladesh)": ["bn-BD-RubaiyatNeural", "bn-BD-KajalNeural"],
    "Hungarian": ["hu-HU-TamasNeural", "hu-HU-NoemiNeural"],
    "Vietnamese": ["vi-VN-HoaiMyNeural", "vi-VN-HuongNeural"],
    "Greek": ["el-GR-AthinaNeural", "el-GR-NestorasNeural"],
    "Danish": ["da-DK-PernilleNeural", "da-DK-MadsNeural"],
    "Hebrew": ["he-IL-AvriNeural", "he-IL-HilaNeural"],
    "Spanish (Spain)": ["es-ES-ElviraNeural", "es-ES-AlvaroNeural"],
    "Spanish (Mexico)": ["es-MX-DaliaNeural", "es-MX-JorgeNeural"],
    "Italian": ["it-IT-ElsaNeural", "it-IT-DiegoNeural"],
    "Chinese (Simplified)": ["zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural"],
    "Korean": ["ko-KR-SunHiNeural", "ko-KR-InJoonNeural"],
    "German": ["de-DE-KatjaNeural", "de-DE-ConradNeural"],
    "Dutch": ["nl-NL-ColetteNeural", "nl-NL-FennaNeural"],
    "Norwegian": ["nb-NO-PernilleNeural", "nb-NO-FinnNeural"],
    "Polish": ["pl-PL-MajaNeural", "pl-PL-JacekNeural"],
    "Portuguese (Brazil)": ["pt-BR-FranciscaNeural", "pt-BR-AntonioNeural"],
    "Portuguese (Portugal)": ["pt-PT-RaquelNeural", "pt-PT-DuarteNeural"],
    "Romanian": ["ro-RO-EmilNeural", "ro-RO-AndreiNeural"],
    "Russian": ["ru-RU-SvetlanaNeural", "ru-RU-DmitryNeural"],
    "Tagalog": ["tl-PH-AngeloNeural", "tl-PH-TessaNeural"],
    "Tamil": ["ta-IN-ValluvarNeural", "ta-IN-KannanNeural"],
    "Thai": ["th-TH-PremwadeeNeural", "th-TH-NiwatNeural"],
    "Turkish": ["tr-TR-AhmetNeural", "tr-TR-EmelNeural"],
    "Ukrainian": ["uk-UA-OstapNeural", "uk-UA-PolinaNeural"],
    "Filipino": ["fil-PH-AngeloNeural", "fil-PH-TessaNeural"],
    "Finnish": ["fi-FI-NooraNeural", "fi-FI-SelmaNeural"],
    "French (Canada)": ["fr-CA-SylvieNeural", "fr-CA-AntoineNeural"],
    "French (France)": ["fr-FR-DeniseNeural", "fr-FR-HenriNeural"],
    "Czech": ["cs-CZ-VlastaNeural", "cs-CZ-AntoninNeural"],
    "Swedish": ["sv-SE-HilleviNeural", "sv-SE-MattiasNeural"],
    "Japanese": ["ja-JP-NanamiNeural", "ja-JP-KeitaNeural"],
}


def update_edge_voices(selected_language):
    voices = edge_voices[selected_language]
    return gr.update(choices=voices, value=voices[0] if voices else None)


def get_folders():
    return sorted(
        (item for item in os.listdir(RVC_MODELS_DIR) if os.path.isdir(os.path.join(RVC_MODELS_DIR, item))),
        key=lambda x: [int(text) if text.isdigit() else text.lower() for text in re.split("([0-9]+)", x)],
    )


def update_models_list():
    return gr.update(choices=get_folders())


def process_file_upload(file):
    return file, gr.update(value=file)


def show_hop_slider(pitch_detection_algo):
    if pitch_detection_algo in ["crepe", "crepe-tiny"]:
        return gr.update(visible=True)
    return gr.update(visible=False)


def swap_visibility():
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(value=""),
        gr.update(value=None),
    )


def swap_buttons():
    return gr.update(visible=False), gr.update(visible=True)
