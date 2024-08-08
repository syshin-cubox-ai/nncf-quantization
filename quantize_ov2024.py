import pathlib

import cv2
import nncf
import numpy as np
import openvino
import torch.utils.data
from torchvision import datasets, transforms


def resize_preserving_aspect_ratio(img: np.ndarray, new_shape: tuple[int, int]) -> np.ndarray:
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    if r != 1:
        interpolation = cv2.INTER_AREA if r < 1 else cv2.INTER_LANCZOS4
        img = cv2.resize(img, None, fx=r, fy=r, interpolation=interpolation)
    return img


def transform_fn(data_item):
    img, _ = data_item
    img = img.numpy().squeeze().transpose((1, 2, 0))

    imgsz = 320
    img = resize_preserving_aspect_ratio(img, (imgsz, imgsz))
    pad = (0, imgsz - img.shape[0], 0, imgsz - img.shape[1])
    img = cv2.copyMakeBorder(img, *pad, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    img = cv2.dnn.blobFromImage(img, 1 / 128, img.shape[:2][::-1], (127.5, 127.5, 127.5))
    return img


def main():
    input_path = pathlib.Path("scrfd/scrfd_500m_bnkps_320.xml")
    output_path = input_path.with_stem(input_path.stem + "_int8")

    model = openvino.Core().read_model(input_path)
    val_dataset = datasets.ImageFolder(
        "D:/data/WIDER_val/images",
        transform=transforms.Compose([transforms.PILToTensor()])
    )
    dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

    calibration_dataset = nncf.Dataset(dataset_loader, transform_fn)
    quantized_model = nncf.quantize(model, calibration_dataset)

    openvino.save_model(quantized_model, output_path)


if __name__ == "__main__":
    main()
