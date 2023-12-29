import os
import random
import shutil

root_path = ".\\train"

new_train_path = ".\\dataset\\train"
new_test_path = ".\\dataset\\test"


images = open('images.txt', 'w')
classes = open('classes.txt', 'w')
image_class_labels = open('image_class_labels.txt', 'w')
train_test_split = open('train_test_split.txt', 'w')

image_id = 1
class_id = 1

dirs = os.listdir(root_path)
for dir in dirs:
    dir_path = os.path.join(root_path, dir)
    if os.path.isdir(dir_path):
        classes.write(f"{class_id} {dir}\n")

        imgs = os.listdir(dir_path)
        for img in imgs:
            img_path = os.path.join(dir_path, img)
            if os.path.isfile(img_path):
                images.write(f"{image_id} {os.path.join(dir, img)}\n")

                image_class_labels.write(f"{image_id} {class_id}\n")
                is_train = random.choices([0, 1], [0.2, 0.8])[0]
                train_test_split.write(f"{image_id} {is_train}\n")
                if is_train:
                    new_path = os.path.join(new_train_path, dir)
                    if not os.path.exists(new_path):
                        os.makedirs(new_path)
                    shutil.copy(img_path, new_path)
                else:
                    new_path = os.path.join(new_test_path, dir)
                    if not os.path.exists(new_path):
                        os.makedirs(new_path)
                    shutil.copy(img_path, new_path)

                image_id += 1

        class_id += 1

images.close()
classes.close()
image_class_labels.close()
train_test_split.close()