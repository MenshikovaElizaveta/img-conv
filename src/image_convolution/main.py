import argparse

from .convolver import apply_convolution
from .utils import load_image, save_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Свертка изображений")

    parser.add_argument("input", help="Входной файл")
    parser.add_argument("output", help="Выходной файл")

    parser.add_argument("--kernel", default="blur", help="Выбор фильтра")
    parser.add_argument("--border", default="constant", help="Тип обработки края")

    args = parser.parse_args()
    
    image = load_image(args.input)

    result = apply_convolution(
        image=image,
        kernel_name=args.kernel,
        border_type=args.border
    )

    save_image(result, args.output)
    print("Обработка завершена успешно!")


if __name__ == "__main__":
    main()
