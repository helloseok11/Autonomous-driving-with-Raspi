from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array
)
import os

# 밝기만 증강
datagen = ImageDataGenerator(
    brightness_range=[0.5, 1.5]
)

# go 학습 이미지 폴더
source_folder = 'source/train_data/right'
save_folder = 'source/train_data/right'

os.makedirs(save_folder, exist_ok=True)

# 기존 증강 이미지를 제외하고 원본 이미지만 선택
image_list = [
    filename
    for filename in os.listdir(source_folder)
    if (
        filename.lower().endswith(('.jpg', '.jpeg', '.png'))
        and not filename.startswith('aug_')
    )
]

print(f"right 원본 이미지 {len(image_list)}개에서 증강 시작")

# 원본 이미지 한 장당 밝기가 다른 이미지 5장 생성
for img_name in image_list:
    img_path = os.path.join(source_folder, img_name)

    img = load_img(img_path)
    arr = img_to_array(img)
    arr = arr.reshape((1,) + arr.shape)

    count = 0

    for _ in datagen.flow(
        arr,
        batch_size=1,
        save_to_dir=save_folder,
        save_prefix='aug_' + os.path.splitext(img_name)[0],
        save_format='jpg',
        seed=42
    ):
        count += 1

        if count >= 5:
            break

print("right 이미지 밝기 증강 완료")