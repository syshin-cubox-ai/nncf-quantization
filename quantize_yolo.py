import pathlib
import shutil
import subprocess

import nncf
import numpy as np
import openvino.runtime
import torch
from openvino.runtime import serialize
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.modules import Detect
from ultralytics.utils import DEFAULT_CFG, LOGGER, colorstr


def get_int8_calibration_dataloader(args, model, prefix=colorstr("OpenVINO:")):
    """Build and return a dataloader suitable for calibration of INT8 models."""
    LOGGER.info(f"{prefix} collecting INT8 calibration images from 'data={args.data}'")
    data = (check_cls_dataset if model.task == "classify" else check_det_dataset)(args.data)
    dataset = YOLODataset(
        data[args.split or "val"],
        data=data,
        task=model.task,
        imgsz=args.imgsz[0],
        augment=False,
        batch_size=1,
    )
    n = len(dataset)
    if n < 300:
        LOGGER.warning(f"{prefix} WARNING ⚠️ >300 images recommended for INT8 calibration, found {n} images.")
    return build_dataloader(dataset, batch=1, workers=0)  # required for batch loading


def transform_fn(data_item) -> np.ndarray:
    """Quantization transform function."""
    data_item: torch.Tensor = data_item["img"] if isinstance(data_item, dict) else data_item
    assert data_item.dtype == torch.uint8, "Input image must be uint8 for the quantization preprocessing"
    im = data_item.numpy().astype(np.float32) / 255.0  # uint8 to fp16/32 and 0 - 255 to 0.0 - 1.0
    return np.expand_dims(im, 0) if im.ndim == 3 else im


def main():
    # Paths
    pt_path = pathlib.Path("runs/pose/train/weights/last.pt")
    data_yaml_path = pathlib.Path("datasets/coco-face-person.yaml")
    xml_path = pathlib.Path(f"temp/{pt_path.stem}.xml")
    output_path = pathlib.Path(f"{pt_path.stem}_int8_openvino_model/{pt_path.stem}.xml")

    # YOLO args
    yolo_model = YOLO(pt_path)
    args = get_cfg(cfg=DEFAULT_CFG)
    args.data = str(data_yaml_path)
    args.batch = 1
    args.imgsz = (640,)

    # Export onnx
    onnx_path = yolo_model.export(format="onnx", simplify=True)
    assert pathlib.Path(onnx_path).is_file()

    # Convert openvino
    subprocess.run(f"mo --input_model {onnx_path} --output_dir {xml_path.parent}", check=True)
    assert xml_path.is_file()

    # Quantize
    core = openvino.runtime.Core()
    ov_model = core.read_model(str(xml_path))

    ignored_scope = None
    if isinstance(yolo_model.model.model[-1], Detect):
        # Includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
        head_module_name = ".".join(list(yolo_model.model.named_modules())[-1][0].split(".")[:2])
        ignored_scope = nncf.IgnoredScope(  # ignore operations
            patterns=[
                f"/{head_module_name}/Add",
                f"/{head_module_name}/Sub",
                f"/{head_module_name}/Mul",
                f"/{head_module_name}/Div",
                f"/{head_module_name}/dfl",
            ],
            names=[f"/{head_module_name}/Sigmoid"],
        )

    quantized_ov_model = nncf.quantize(
        model=ov_model,
        calibration_dataset=nncf.Dataset(get_int8_calibration_dataloader(args, yolo_model), transform_fn),
        preset=nncf.QuantizationPreset.MIXED,
        ignored_scope=ignored_scope,
    )
    serialize(quantized_ov_model, str(output_path))

    # Remove useless files
    shutil.rmtree(xml_path.parent)


if __name__ == "__main__":
    main()
