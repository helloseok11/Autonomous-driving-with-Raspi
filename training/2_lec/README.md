# CNN 학습 코드와 모델

강의 코드 `AI_CODE/2_lec`에서 사용한 파일을 보관합니다. 원본 Python 코드와 H5 모델은 그대로 복사했습니다.

## 파일

| 파일 | 역할 |
| --- | --- |
| `1_split.py` | 이미지 분할: 70% 학습, 15% 검증, 나머지 테스트; seed 42 |
| `2_augment.py` | 밝기 0.5~1.5 범위, 원본당 5장 증강 |
| `3. CNN_train.py` | 64×64 입력, Conv2D·MaxPool 2단계, Dense 64, softmax, Adam, 10 epochs |
| `4_lane_cnn.py` | 웹캠 하단 ROI의 방향 분류 |
| `cnn_model_gen_noaug.h5` | 웹캠 예제가 기본으로 읽는 모델 |
| `cnn_model_gen_noaug_2.h5` | 학습 코드의 출력 파일명에 해당하는 모델 |

## 실행 기준 경로

원본 상대 경로를 유지하기 위해 **저장소의 `training/` 폴더에서 실행**합니다. 의존성은 TensorFlow/Keras, NumPy, OpenCV입니다. 학습 당시의 정확한 버전은 기록이 없어 고정하지 않았습니다.

```text
training/
├── 2_lec/
└── source/
    ├── data/{go,left,right}/
    ├── train_data/{go,left,right}/
    ├── val_data/{go,left,right}/
    └── test_data/{go,left,right}/
```

`source/`의 실제 이미지는 포함하지 않았습니다. `1_split.py`의 `class_name` 기본값은 `right`이므로 `go`, `left`, `right` 각각에 대해 설정하고 실행해야 합니다. `2_augment.py`도 `right` 경로가 기본값이며, 다른 클래스를 증강하려면 입력·출력 경로를 함께 변경합니다. 데이터 분할 후 학습 데이터에만 증강을 적용합니다.

```bash
cd training
python "2_lec/1_split.py"
# 필요할 때 클래스별 경로를 설정한 뒤 실행
python "2_lec/2_augment.py"
python "2_lec/3. CNN_train.py"
python "2_lec/4_lane_cnn.py"
```

학습 코드의 실시간 제너레이터는 정규화만 수행합니다. 앞 단계에서 디스크에 생성한 증강 이미지가 있다면 그 이미지도 학습에 포함되므로 `noaug` 파일명만으로 증강 여부를 판단하지 않습니다.

학습은 `cnn_model_gen_noaug_2.h5`로 저장하지만 추론은 `cnn_model_gen_noaug.h5`를 읽습니다. 새 모델을 시험하려면 추론 코드의 `load_model()` 경로를 변경합니다. 학습 시 출력되는 클래스 인덱스가 추론의 `['go', 'left', 'right']`와 같은지도 확인합니다.

원본 추론은 OpenCV의 BGR 영상을 사용하고 학습 이미지 로더는 RGB를 사용합니다. ROI 좌표도 640×480 프레임을 가정합니다. 모델을 비교하거나 재사용할 때 색상 순서·ROI·입력 해상도를 학습 전처리와 맞춰 확인해야 합니다.

두 H5 파일 중 최종 CNN 모델이 어느 것인지는 별도로 확정되지 않았습니다. 최종 차량 주행 구성은 OpenCV+YOLO이며, 이 폴더는 CNN 학습·실험 기록입니다.
