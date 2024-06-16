# preprocess_data.py
"""
    @description    :统一图像尺寸
"""
import os
from PIL import Image
from torchvision import transforms
from config import config


def preprocess_images(input_dir, output_dir, image_size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])

    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)

        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image = Image.open(image_path).convert('RGB')
            image = transform(image)
            output_image_path = os.path.join(output_class_dir, image_name)
            transforms.ToPILImage()(image).save(output_image_path)
            print(f'Processed {output_image_path}')


if __name__ == "__main__":
    preprocess_images(config.DATA_DIR, config.PROCESSED_DATA_DIR, config.IMAGE_SIZE)