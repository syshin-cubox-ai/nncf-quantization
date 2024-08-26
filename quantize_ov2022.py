import pathlib

import nncf
import openvino.runtime
import torch.utils.data
from torchvision import datasets, transforms


def transform_fn(data_item):
    img, _ = data_item
    return img


def main():
    input_path = pathlib.Path("C:/Users/synml/Desktop/qmagnet.xml")
    output_path = input_path.with_stem(input_path.stem + "_int8")
    dataset_root = "D:/data/fiqa"
    subset_size = 300
    fast_bias_correction = True

    core = openvino.runtime.Core()
    ov_model = core.read_model(str(input_path))
    val_dataset = datasets.ImageFolder(
        dataset_root,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

    calibration_dataset = nncf.Dataset(dataset_loader, transform_fn)
    quantized_ov_model = nncf.quantize(ov_model, calibration_dataset,
                                       subset_size=subset_size, fast_bias_correction=fast_bias_correction)

    openvino.runtime.serialize(quantized_ov_model, str(output_path))


if __name__ == "__main__":
    main()