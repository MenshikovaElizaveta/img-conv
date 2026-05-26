import typer

from .convolver import apply_convolution
from .utils import load_image, save_image

app = typer.Typer()


@app.command()
def main(
    input_path: str,
    output_path: str,
    kernel: str = "blur",
    border: str = "constant",
) -> None:
    """
    Свертка изображений.
    """

    image = load_image(input_path)

    result = apply_convolution(
        image=image,
        kernel_name=kernel,
        border_type=border,
    )

    save_image(result, output_path)

    print("Обработка завершена успешно!")


if __name__ == "__main__":
    app()
