import subprocess
from pathlib import Path

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
    data = (check_cls_dataset if model.task == "classify" else check_det_dataset)(
        args.data
    )
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
        LOGGER.warning(
            f"{prefix} WARNING ⚠️ >300 images recommended for INT8 calibration, found {n} images."
        )
    return build_dataloader(dataset, batch=1, workers=0)  # required for batch loading


def transform_fn(data_item) -> np.ndarray:
    """Quantization transform function."""
    data_item: torch.Tensor = (
        data_item["img"] if isinstance(data_item, dict) else data_item
    )
    assert data_item.dtype == torch.uint8, (
        "Input image must be uint8 for the quantization preprocessing"
    )
    im = (
        data_item.numpy().astype(np.float32) / 255.0
    )  # uint8 to fp16/32 and 0 - 255 to 0.0 - 1.0
    return np.expand_dims(im, 0) if im.ndim == 3 else im


def main():
    # Args
    pt_path = Path("runs/detect/bezel_detector/weights/last.pt").resolve(strict=True)
    data_yaml_path = Path(
        "../../data/home-collected-bezel-background-v5/data.yaml"
    ).resolve(strict=True)
    subset_size = 300
    fast_bias_correction = False

    # Paths
    onnx_path = pt_path.with_suffix(".onnx")
    xml_path = Path(f"{pt_path.stem}_openvino_model/{pt_path.stem}.xml")
    output_path = Path(f"{pt_path.stem}_int8_openvino_model/{pt_path.stem}.xml")

    # YOLO args
    yolo_model = YOLO(pt_path)
    args = get_cfg(cfg=DEFAULT_CFG)
    args.data = str(data_yaml_path)
    args.batch = 1
    args.imgsz = (640,)

    # Export onnx
    if not onnx_path.is_file():
        yolo_model.export(format="onnx", opset=17)
        print("\n!!Please re-run due to a forced termination bug!!")
        exit(0)
    assert Path(onnx_path).is_file()

    # Convert openvino
    subprocess.run(
        f"mo --input_model {onnx_path} --output_dir {xml_path.parent} --compress_to_fp16",
        check=True,
        shell=True,
    )
    assert xml_path.is_file()

    # Read model
    core = openvino.runtime.Core()
    ov_model = core.read_model(str(xml_path))

    # Make ignored scope
    ignored_scope = None
    if isinstance(yolo_model.model.model[-1], Detect):
        # Includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
        head_module_name = ".".join(
            list(yolo_model.model.named_modules())[-1][0].split(".")[:2]
        )
        ignored_scope = nncf.IgnoredScope(  # ignore operations
            patterns=[
                f"/{head_module_name}/Add",
                f"/{head_module_name}/Sub",
                f"/{head_module_name}/Mul",
                f"/{head_module_name}/Div",
                f"/{head_module_name}/dfl",
            ],
            types=["Sigmoid"],
        )

    # Quantize
    quantized_ov_model = nncf.quantize(
        model=ov_model,
        calibration_dataset=nncf.Dataset(
            get_int8_calibration_dataloader(args, yolo_model), transform_fn
        ),
        preset=nncf.QuantizationPreset.MIXED,
        ignored_scope=ignored_scope,
        subset_size=subset_size,
        fast_bias_correction=fast_bias_correction,
    )
    serialize(quantized_ov_model, str(output_path))

    # Remove useless files
    xml_path.with_suffix(".mapping").unlink()


if __name__ == "__main__":
    main()
