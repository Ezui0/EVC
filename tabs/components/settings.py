import gradio as gr


def settings():
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
                            info="Smaller values lead to longer conversions, which increases the risk of artifacts in the voice, but achieves more accurate pitch reproduction.",
                            interactive=True,
                            visible=False,
                        )
                        index_rate = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.1,
                            value=0,
                            label="Index influence",
                            info="The influence of the index file; the higher the value, the greater its effect. However, choosing lower values can help mitigate artifacts present in the audio.",
                            interactive=True,
                            visible=True,
                        )
                        volume_envelope = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.01,
                            value=1,
                            label="RMS mix rate",
                            info="Replace or blend with the volume envelope of the output signal. The closer the value is to 1, the more of the output envelope is used.",
                            interactive=True,
                            visible=True,
                        )
                        protect = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            step=0.01,
                            value=0.5,
                            label="Consonant protection",
                            info="Protect consonants and breathing sounds to avoid electroacoustic breaks and artifacts. The maximum value of 0.5 provides full protection. Lowering this value may reduce protection but also lessen the indexing effect.",
                            interactive=True,
                            visible=True,
                        )

            with gr.Accordion("Advanced Settings", open=False):
                with gr.Column():
                    with gr.Row():
                        f0_min = gr.Slider(
                            minimum=1,
                            maximum=120,
                            step=1,
                            value=50,
                            label="Minimum pitch range",
                            info="Defines the lower bound of the pitch range the algorithm will use to determine the fundamental frequency (F0) in the audio signal.",
                            interactive=True,
                            visible=True,
                        )
                        f0_max = gr.Slider(
                            minimum=380,
                            maximum=16000,
                            step=1,
                            value=1100,
                            label="Maximum pitch range",
                            info="Defines the upper bound of the pitch range the algorithm will use to determine the fundamental frequency (F0) in the audio signal.",
                            interactive=True,
                            visible=True,
                        )

    return f0_method, hop_length, index_rate, volume_envelope, protect, f0_min, f0_max
