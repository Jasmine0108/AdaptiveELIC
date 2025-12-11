import os
import random
import json

image_folder = "../dataset/Screen-Content"

image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]

all_images = [
    f for f in os.listdir(image_folder)
    if os.path.isfile(os.path.join(image_folder, f)) and os.path.splitext(f)[1].lower() in image_extensions
]
# Choose 25 images for training
train_images = random.sample(all_images, 25)

test_images = [f for f in all_images if f not in train_images]

with open("../splits/Screen-Content/train.json", "w") as f:
    json.dump(train_images, f, indent=4)

with open("../splits/Screen-Content/inference.json", "w") as f:
    json.dump(test_images, f, indent=4)

print(f"total: {len(all_images)}")
print(f"train images: {len(train_images)}")
print(f"inference images: {len(test_images)}")
print("Filenames have been written to train.json and inference.json")