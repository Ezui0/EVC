from assets.logging_config import configure_logging
from assets.model_installer import check_and_install_models

configure_logging(True, False, "WARNING")
check_and_install_models()

import sys
from typing import Any

import gradio as gr
from PolUVR.utils import PolUVR_UI

from tabs.inference import inference_tab

DEFAULT_SERVER_NAME = "127.0.0.1"
DEFAULT_PORT = 4000
MAX_PORT_ATTEMPTS = 10

output_message_component = output_message()



with gr.Blocks(
    title="EVC" if not is_offline_mode() else "EVC (offline)",
    css="footer{display:none !important}",
    theme=gr.themes.Base(
        primary_hue="green",
        secondary_hue="green",
        neutral_hue="neutral",
        spacing_size="sm",
        radius_size="lg",
    ),
) as app:

    
    with gr.Tab("Inference"):
        inference_tab()

    with gr.Tab("PolUVR (UVR)"):
        PolUVR_UI("models/UVR_models", "output/UVR_output")

    
def launch_gradio(server_name: str, server_port: int) -> None:
    app.launch(
        favicon_path="assets/logo.ico",
        share="--share" in sys.argv,
        inbrowser="--open" in sys.argv,
        server_name=server_name,
        server_port=server_port,
        show_error=True,
    )


def get_value_from_args(key: str, default: Any = None) -> Any:
    if key in sys.argv:
        index = sys.argv.index(key) + 1
        if index < len(sys.argv):
            return sys.argv[index]
    return default


if __name__ == "__main__":
    port = int(get_value_from_args("--port", DEFAULT_PORT))
    server = get_value_from_args("--server-name", DEFAULT_SERVER_NAME)

    for _ in range(MAX_PORT_ATTEMPTS):
        try:
            launch_gradio(server, port)
            break
        except OSError:
            print(f"Не удалось запустить на порту {port}, повторите попытку на порту {port - 1}...")
            port -= 1
        except Exception as error:
            print(f"Произошла ошибка при запуске Gradio: {error}")
            break
