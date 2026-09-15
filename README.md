# Raspberry Pi 자율주행 프로젝트

자율주행 모형차 경진대회를 준비하며 만든 주행 코드, Raspberry Pi 보드의 베이스라인 예제, CNN·YOLO 학습 코드와 모델을 정리한 저장소입니다.

**최종 주행은 OpenCV 차선 인식 + YOLO 객체 인식으로 구성합니다.** CNN은 `go / left / right` 방향 분류를 학습하고 실험한 과정으로 함께 보관합니다.

## 저장소 구성

| 경로 | 내용 |
| --- | --- |
| [driving/](driving/) | 최종 주행 코드 및 OpenCV 단독 주행 코드 |
| [raspi/](raspi/) | Raspberry Pi 보드에 있던 카메라·LED·조향·모터·촬영·CNN·YOLO 베이스라인 |
| [training/2_lec/](training/2_lec/) | CNN 데이터 분할·증강·학습·웹캠 추론 코드와 H5 모델 2개 |
| [training/3_lec/](training/3_lec/) | YOLO 학습 노트북·Python 코드·웹캠 추론 예제와 PT 모델 |

## 최종 주행

실행 파일: [`driving/OpenCV+YOLO.py`](driving/OpenCV%2BYOLO.py)

- **OpenCV:** 카메라 하단 ROI에서 흰색 차선을 추출하고, Hough 선분으로 차선 중심을 계산해 조향합니다. 차선 중심과 조향 변화량을 완화하고 곡선·차선 유실 상황에 따라 속도를 조절합니다.
- **YOLO:** 별도 스레드에서 객체를 탐지하고 사람 정지, 신호등 정지·방향 전환, 차량 회피 동작을 처리합니다.
- **차량 제어:** `afb1`의 카메라·GPIO 인터페이스로 모터와 서보를 제어합니다.
- [`driving/opencv_only.py`](driving/opencv_only.py)는 OpenCV 단독 주행 파일입니다.

주행 코드가 사용하는 객체 이름은 `circle`, `person`, `car1`, `car2`, `red light`, `left`, `right`입니다. 이는 **코드가 기대하는 이름**이며, 실제 모델의 클래스 이름과 일치하는지 실행 전에 확인해야 합니다.

### 최종 YOLO 모델 연결

최종 모델은 [`best-신호등쪽 수정.pt`](training/3_lec/best-신호등쪽%20수정.pt)입니다. 다른 `best*.pt`는 이전 실험 모델로 보관합니다.

현재 주행 코드는 실행 파일과 같은 폴더의 `best.pt`를 읽습니다. Raspberry Pi에서 저장소 루트를 기준으로 다음과 같이 복사합니다.

```bash
cp "training/3_lec/best-신호등쪽 수정.pt" "driving/best.pt"
python3 "driving/OpenCV+YOLO.py"
```

Raspberry Pi에 카메라·모터·서보와 해당 보드용 `afb1` 환경이 준비되어 있어야 합니다. Python 의존성은 `numpy`, `opencv-python`, `ultralytics`이며, 보드의 기존 OpenCV·카메라 환경과 호환되는 구성을 사용합니다.

현재 코드는 시작 직후 `STARTUP_SPEED=100`으로 `STARTUP_SEC=5.0`초간 전진한 뒤 일반 주행 루프에 진입합니다. 차량을 실행하기 전에 이 값과 서보 각도·회피 시간을 실제 차량 및 트랙에 맞게 확인합니다.

## CNN 학습과 실험

원본 위치는 강의 코드의 `AI_CODE/2_lec`입니다. 파일별 설명과 실행 순서는 [CNN 안내](training/2_lec/README.md)를 참고합니다.

1. `1_split.py`: 클래스별 이미지를 학습 70%·검증 15%·테스트 나머지로 분할합니다.
2. `2_augment.py`: 학습 이미지의 밝기를 변경해 원본당 5장을 생성합니다. 필요한 경우에만 수행합니다.
3. `3. CNN_train.py`: 64×64 RGB 이미지를 정규화하고 2개 합성곱 층을 가진 CNN을 10 epochs 학습합니다.
4. `4_lane_cnn.py`: 웹캠 영상의 하단 ROI로 `go / left / right`를 예측합니다.

| 모델 | 구분 |
| --- | --- |
| `cnn_model_gen_noaug.h5` | 현재 웹캠 추론 예제가 참조하는 모델 |
| `cnn_model_gen_noaug_2.h5` | 현재 학습 스크립트의 저장 파일명과 일치하는 모델 |

두 모델의 우열이나 최종 CNN 모델 여부는 파일명만으로 단정하지 않습니다. 최종 OpenCV+YOLO 주행 파일은 CNN 모델을 사용하지 않습니다.

## YOLO 학습과 실험

원본 위치는 강의 코드의 `AI_CODE/3_lec`입니다. 자세한 내용은 [YOLO 안내](training/3_lec/README.md)를 참고합니다.

- `YOLO.ipynb`: Colab에서 사용한 학습 노트북입니다.
- `yolo.py`: 노트북에서 내보낸 Python 학습 코드입니다.
- 노트북에는 `yolov8n.pt`, `yolo11n.pt`, `yolo26n.pt`를 사용하는 개별 실험 셀이 있습니다. 필요한 모델의 셀을 선택해 실행합니다.
- 각 학습 셀의 설정은 `epochs=30`, `imgsz=640`, `batch=16`이며, `data.yaml`의 `test` 분할을 평가합니다.
- `1_yolo_cam.py`, `2_yolo_cam_fps.py`: 웹캠 추론 및 FPS 확인용 예제입니다. 원본 예제는 이전 모델을 참조하므로 최종 모델을 확인할 때 경로를 변경해야 합니다.

업로드용 학습 코드에서는 Roboflow 다운로드 키와 개인 Colab 링크를 제거하고, 노트북 실행 출력도 비웠습니다. 학습 로직은 보존했습니다.

## 데이터와 재현 범위

이 저장소에 추가한 자료는 학습·추론 코드와 저장된 모델 파일입니다. 원본 학습 이미지, YOLO 라벨, 실제 학습에 사용한 `data.yaml`은 이번 추가 항목에 포함하지 않았습니다. 재학습하려면 해당 데이터와 분할 구성을 별도로 준비해야 합니다.

학습 당시의 패키지 버전, 최종 모델과 각 실험 셀의 정확한 대응, 검증된 정확도·mAP·실차 성능 수치는 현재 자료만으로 확정하지 않았습니다. 모델 파일명이나 코드의 설정을 실제 측정 결과로 해석하지 않습니다.

강의에서 제공된 코드·베이스라인과 이를 이용한 프로젝트 결과를 함께 보관합니다. 외부 코드·데이터의 출처와 이용 조건은 각 원 출처를 따릅니다.
