# checks if the datasets match (pics have labels, labels have pics)

import os

def get_filenames(directory, extensions):
    extensions = {ext.lower() for ext in extensions}
    return set(os.path.splitext(f)[0] for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in extensions)

def check_images_labels(images_dir, labels_dir, image_exts=['.jpg', '.jpeg', '.png', 'webp'], label_ext='.txt'):
    image_filenames = get_filenames(images_dir, image_exts)
    label_filenames = get_filenames(labels_dir, [label_ext])
    
    images_without_labels = image_filenames - label_filenames
    labels_without_images = label_filenames - image_filenames
    
    if images_without_labels:
        print("Images without corresponding labels:")
        for img in images_without_labels:
            print(f"{img} (any of {image_exts})")
    
    if labels_without_images:
        print("Labels without corresponding images:")
        for lbl in labels_without_images:
            print(f"{lbl}{label_ext}")
    
    if not images_without_labels and not labels_without_images:
        print("All images have corresponding labels, and all labels have corresponding images.")

images_directory = "C:/Users/Juls/Desktop/dataset(9-23)/images/train"
labels_directory = "C:/Users/Juls/Desktop/dataset(9-23)/labels/train"

check_images_labels(images_directory, labels_directory)
