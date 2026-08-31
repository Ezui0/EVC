import sys
from typing import Any
import gradio as gr
from tabs.inference import inference_tab
from tabs.download import dltabs
from assets.logging_config import configure_logging
from assets.model_installer import check_and_install_models

configure_logging(True, False, "WARNING")
# The --offline flag (used by run-PolGen.sh) skips model downloads at startup
check_and_install_models(offline="--offline" in sys.argv)



DEFAULT_SERVER_NAME = "127.0.0.1"
DEFAULT_PORT = 4000
MAX_PORT_ATTEMPTS = 10




with gr.Blocks(
    title="EVC",
    css="footer{display:none !important}",
    theme=gr.themes.Base(
        primary_hue="green",
        secondary_hue="green",
        neutral_hue="neutral",
        spacing_size="sm",
        radius_size="lg",
    ),
) as app:

    with gr.Tabs():
        inference_tab()
    with gr.Tabs():
        dltabs()

    
def launch_gradio(server_name: str, server_port: int) -> None:
    app.launch(
        share="--share" in sys.argv,
        inbrowser="--open" in sys.argv,
        server_name=server_name,
        server_port=server_port,
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
            print(f"Failed to start on port {port}, retrying on port {port + 1}...")
            port += 1
        except Exception as error:
            print(f"Error launching Gradio: {error}")
            break
