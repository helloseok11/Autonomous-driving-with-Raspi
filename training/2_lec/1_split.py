import os
import shutil
import random

source_dir = 'source/data'
class_name = 'right'
class_path = os.path.join(source_dir, class_name)

train_dir = 'source/train_data'
val_dir = 'source/val_data'
test_dir = 'source/test_data'

train_ratio = 0.7
val_ratio = 0.15

random.seed(42)

images = [
    filename
    for filename in os.listdir(class_path)
    if filename.lower().endswith(('.jpg', '.jpeg', '.png'))
]

random.shuffle(images)

total = len(images)
train_end = int(total * train_ratio)
val_end = train_end + int(total * val_ratio)

split_data = {
    train_dir: images[:train_end],
    val_dir: images[train_end:val_end],
    test_dir: images[val_end:]
}

for target_dir, split_images in split_data.items():
    output_dir = os.path.join(target_dir, class_name)
    os.makedirs(output_dir, exist_ok=True)

    for image in split_images:
        shutil.copy2(
            os.path.join(class_path, image),
            os.path.join(output_dir, image)
        )

print(
    f"{class_name} 분리 완료: "
    f"train={train_end}, "
    f"val={val_end - train_end}, "
    f"test={total - val_end}"
)