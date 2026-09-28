"""Colab previews and explicit download buttons for generated HTML results."""
from pathlib import Path


def download_button(path):
    import ipywidgets as widgets
    from IPython.display import display
    from google.colab import files

    path = Path(path)
    button = widgets.Button(description="HTML 다운로드", icon="download")
    status = widgets.Output()

    def download(_):
        with status:
            status.clear_output()
            if not path.is_file():
                raise FileNotFoundError(path)
            files.download(str(path))

    button.on_click(download)
    display(button, status)


def show_validation_results(directory, case_ids):
    import ipywidgets as widgets
    from IPython.display import display
    from visualize_validation import build_figure, load_comparison

    directory = Path(directory)
    paths = [directory / f"comparison_{case_id}.npz" for case_id in case_ids]
    if not paths:
        return
    chooser = widgets.Dropdown(options=[(p.stem.removeprefix("comparison_"), str(p)) for p in paths],
                               description="검증 케이스:", layout=widgets.Layout(width="90%"))
    view = widgets.ToggleButtons(options=[("회전 3D", "3d"), ("전개면 2D", "2d")])
    output = widgets.Output()
    button = widgets.Button(description="HTML 다운로드", icon="download")
    status = widgets.Output()

    def render(*_):
        with output:
            output.clear_output(wait=True)
            figure = build_figure(**load_comparison(chooser.value), view=view.value)
            figure.show(renderer="colab")

    def download(_):
        from google.colab import files
        with status:
            status.clear_output()
            files.download(str(Path(chooser.value).with_suffix(".html")))

    chooser.observe(render, names="value")
    view.observe(render, names="value")
    button.on_click(download)
    display(chooser, view, button, status, output)
    render()
