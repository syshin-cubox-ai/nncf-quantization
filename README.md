# 🔎 개요
이 리포지토리는 nncf 구버전을 기반으로 openvino 2022의 int8 양자화를 수행합니다.

양자화 과정은 아래와 같습니다.

PyTorch → ONNX → OpenVINO FP16 → OpenVINO INT8

PyTorch 또는 ONNX에서 바로 OpenVINO INT8로 변환하지 않는 이유는 openvino 2022 버전에서 오류가 발생하기 때문입니다.

따라서 mo를 사용하여 먼저 OpenVINO IR로 변환하고, 이를 양자화해야 문제가 없습니다.


# 🛠️ 환경 설정
`create_conda_env.txt`를 전체 복사하여 쉘에 붙여넣습니다.

그러면 conda 가상환경이 만들어지고, 앞으로 이것을 사용합니다.


# 🎯 실행
양자화에 사용되는 arguments를 설명합니다.

- subset_size: calibration에 사용할 이미지 개수
- fast_bias_correction: 속도는 빠르지만 정확도가 낮은 보정 방법을 사용할 것인지 여부,  
  `False`로 설정하면 느리지만 더 정확한 보정 방법을 사용합니다.


## 일반적인 모델 양자화
`quantize_ov2022.py`는 범용적으로 사용 가능한 양자화 코드입니다.

아래 순서로 진행합니다.

1. calibration에 사용할 이미지들을 
[torchvision.datasets.ImageFolder](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html) 형식으로 배치합니다.

2. input_path 등의 arguments를 적절히 설정하고, transform_fn()에 이미지 전처리 과정을 추가합니다.

3. 다음 명령을 실행합니다: `python quantize_ov2022.py`

4. 양자화된 모델은 input_path의 같은 폴더에 저장되어 있습니다.


## YOLO 기반 모델 양자화
`quantize_yolo.py`는 ultralytics 패키지의 YOLO 기반 모델을 양자화하는 코드입니다.

ultralytics 패키지는 최신 버전의 openvino만 양자화를 지원하여 이 코드가 반드시 필요합니다.

1. datasets 폴더에 ultralytics에서 사용하는 데이터셋 cfg yaml 파일을 넣어둡니다.

2. pt 파일을 리포지토리 경로에 넣어둡니다. runs 폴더 째로 넣어두어도 됩니다.

3. 다음 명령을 실행합니다: `python quantize_yolo.py`

4. 양자화된 모델은 리포지토리 경로에 있습니다.  
   `_openvino_model`은 FP16, `_int8_openvino_model`은 INT8입니다.