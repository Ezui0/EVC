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

# Определяем пути к папкам и файлам (константы)
RVC_MODELS_DIR = os.path.join(os.getcwd(), "models", "RVC_models")
OUTPUT_DIR = os.path.join(os.getcwd(), "output", "RVC_output")
HUBERT_BASE_PATH = os.path.join(os.getcwd(), "rvc", "models", "embedders", "hubert_base.pt")

# Создаем папки, если их нет
os.makedirs(RVC_MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Инициализация конфигурации
config = Config()

# Кэш загруженной модели Hubert, чтобы не перезагружать её на каждую конвертацию
_hubert_model = None


# Отображает прогресс выполнения задачи.
def display_progress(percent, message, progress=gr.Progress()):
    progress(percent, desc=message)


def print_display_progress(percent, message, progress=gr.Progress()):
    print(message)
    progress(percent, desc=message)


# Загружает модель RVC и индекс по имени модели.
def load_rvc_model(rvc_model):
    # Формируем путь к директории модели
    model_dir = os.path.join(RVC_MODELS_DIR, rvc_model)
    if not os.path.isdir(model_dir):
        raise gr.Error(
            f"\033[91mОШИБКА!\033[0m Модель {rvc_model} не обнаружена. Возможно, вы допустили ошибку в названии или указали неверную ссылку при установке."
        )

    # Получаем список файлов в директории модели
    model_files = os.listdir(model_dir)

    # Находим файл модели с расширением .pth
    rvc_model_path = next((os.path.join(model_dir, f) for f in model_files if f.endswith(".pth")), None)
    # Находим файл индекса с расширением .index
    rvc_index_path = next((os.path.join(model_dir, f) for f in model_files if f.endswith(".index")), None)

    # Проверяем, существует ли файл модели
    if not rvc_model_path:
        raise gr.Error(
            f"\033[91mОШИБКА!\033[0m Модель {rvc_model} не обнаружена. Возможно, вы допустили ошибку в названии или указали неверную ссылку при установке."
        )

    return rvc_model_path, rvc_index_path


# Загружает модель Hubert (с кэшированием между запросами)
def load_hubert(model_path):
    global _hubert_model

    if _hubert_model is None:
        hubert = fairseq.load_model(model_path)
        hubert = hubert.to(config.device).float()
        hubert.eval()
        _hubert_model = hubert

    return _hubert_model


# Получает конвертер голоса
def get_vc(model_path):
    # Загружаем состояние модели из файла (fallback для нестандартных чекпоинтов)
    try:
        cpt = torch.load(model_path, map_location="cpu", weights_only=True)
    except Exception:
        cpt = torch.load(model_path, map_location="cpu", weights_only=False)

    # Проверяем корректность формата модели
    if "config" not in cpt or "weight" not in cpt:
        raise gr.Error(f"Некорректный формат для {model_path}. Используйте голосовую модель, обученную на RVC v2.")

    # Извлекаем параметры модели
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    pitch_guidance = bool(cpt.get("f0", 1))
    version = cpt.get("version", "v1")

    # vocoder = cpt.get("vocoder", "HiFi-GAN") — на будущее
    input_dim = 768 if version == "v2" else 256

    # Инициализируем синтезатор
    net_g = Synthesizer(*cpt["config"], use_f0=pitch_guidance, input_dim=input_dim)

    # Удаляем ненужный слой
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g = net_g.to(config.device).float()
    net_g.eval()

    # Инициализируем объект конвертера голоса
    vc = VC(tgt_sr, config)
    return cpt, version, net_g, tgt_sr, vc


# Конвертируем аудио в стерео и выбранный пользователем формат
def convert_audio(input_audio, output_audio, output_format):
    # Загружаем аудиофайл
    audio = AudioSegment.from_file(input_audio)

    # Если аудио моно, конвертируем его в стерео
    if audio.channels == 1:
        audio = audio.set_channels(2)

    # Формат по умолчанию — берётся из расширения выходного файла
    export_format = output_format or os.path.splitext(output_audio)[1].lstrip(".")
    export_format = export_format.lower()

    # Сохраняем аудиофайл в выбранном формате
    audio.export(output_audio, format=export_format)


# Синтезирует текст в речь с использованием edge_tts.
async def text_to_speech(voice, text, rate, volume, pitch, output_path):
    if not -100 <= rate <= 100:
        raise ValueError("Rate должен быть в диапазоне от -100% до +100%")
    if not -100 <= volume <= 100:
        raise ValueError("Volume должен быть в диапазоне от -100% до +100%")
    if not -100 <= pitch <= 100:
        raise ValueError("Pitch должен быть в диапазоне от -100Hz до +100Hz")

    rate = f"+{rate}%" if rate >= 0 else f"{rate}%"
    volume = f"+{volume}%" if volume >= 0 else f"{volume}%"
    pitch = f"+{pitch}Hz" if pitch >= 0 else f"{pitch}Hz"

    communicate = edge_tts.Communicate(voice=voice, text=text, rate=rate, volume=volume, pitch=pitch)
    await communicate.save(output_path)


# Основной конвейер конвертации: возвращает путь к готовому файлу.
# Порядок параметров совпадает с порядком входов gr.Button.click в tabs/inference.py.
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
        raise gr.Error("Выберите модель голоса для преобразования.")
    if not input_path:
        raise gr.Error("Выберите или загрузите аудиофайл для преобразования.")
    if not os.path.exists(input_path):
        raise gr.Error(
            f"Не удалось найти файл '{input_path}'. Убедитесь, что он загрузился или проверьте правильность пути к нему."
        )

    output_format = (str(output_format) or "wav").lower().lstrip(".")

    print_display_progress(0, "\n[⚙️] Запуск конвейера генерации...")

    # Загружаем модель Hubert
    display_progress(0.1, "Загружаем модель Hubert...")
    hubert_model = load_hubert(HUBERT_BASE_PATH)
    # Загружаем модель RVC и индекс
    display_progress(0.2, "Загружаем модель RVC и индекс...")
    model_path, index_path = load_rvc_model(rvc_model)
    # Получаем конвертер голоса
    display_progress(0.3, "Получаем конвертер голоса...")
    cpt, version, net_g, tgt_sr, vc = get_vc(model_path)
    pitch_guidance = bool(cpt.get("f0", 1))

    # Построение имени выходного файла
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    if len(base_name) > 50:
        gr.Warning("Имя файла превышает 50 символов и будет сокращено для удобства использования.")
        base_name = "Made_in_PolGen"  # Сменить имя файла, если длина исходного более 50 символов
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}_({rvc_model}).{output_format}")

    # Загружаем аудиофайл
    display_progress(0.4, "Загружаем аудиофайл...")
    audio = load_audio(input_path, 16000)

    print_display_progress(0.5, f"[🌌] Преобразование аудио — {base_name}...")
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

    # Сохраняем результат во временный wav-файл, затем экспортируем
    # в выбранный пользователем формат с правильным расширением.
    display_progress(0.6, "Сохраняем результат...")
    tmp_fd, tmp_wav_path = tempfile.mkstemp(prefix="polgen_", suffix=".wav", dir=OUTPUT_DIR)
    os.close(tmp_fd)
    try:
        wavfile.write(tmp_wav_path, tgt_sr, audio_opt)

        # Конвертируем файл в стерео и выбранный пользователем формат
        print_display_progress(0.8, "[💫] Конвертация аудио в стерео...")
        convert_audio(tmp_wav_path, output_path, output_format)
    finally:
        if os.path.exists(tmp_wav_path):
            os.remove(tmp_wav_path)

    # Освобождаем память
    display_progress(0.9, "Освобождаем память...")
    del cpt, net_g, vc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print_display_progress(1.0, f"[✅] Преобразование завершено — {output_path}")
    return output_path


# Конвертация одиночного аудиофайла через GUI.
# Возвращает (сообщение, аудиокомпонент) под outputs=[vc_output1, vc_output2].
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
    f0_file=None,  # Зарезервировано: файл кривой F0 пока не поддерживается
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
    message = f"[✅] Преобразование завершено — {os.path.basename(output_path)}"
    return message, gr.Audio(output_path, label=os.path.basename(output_path))


# Конвертация батча папки/файлов; возвращает только текстовое сообщение
# под outputs=[vc_output3].
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
        raise gr.Error("Введите необходимый текст в поле для ввода.")
    if not tts_voice:
        raise gr.Error("Выберите язык и голос для синтеза речи.")

    display_progress(0.2, "[🎙️] Синтезируем речь...")
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
