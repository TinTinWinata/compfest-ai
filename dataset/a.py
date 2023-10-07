import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Assuming the csv file is in the same directory as this script
metadata_filepath = './HAM10000_metadata.csv'
images_directory_1 = './dcnn-dataset/skin-lesion/HAM10000_images_part_1'
images_directory_2 = './dcnn-dataset/skin-lesion/HAM10000_images_part_2'

# Read the metadata
metadata_df = pd.read_csv(metadata_filepath)

# Mapping abbreviations to full titles
lesion_type_map = {
    "akiec": "Actinic keratoses / Bowen's disease",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions"
}

def find_image_path(filename):
    """Helper function to search for an image in both directories"""
    if os.path.exists(os.path.join(images_directory_1, filename)):
        return os.path.join(images_directory_1, filename)
    elif os.path.exists(os.path.join(images_directory_2, filename)):
        return os.path.join(images_directory_2, filename)
    else:
        return None

# For each lesion type, pick one image and display it
plt.figure(figsize=(8, 8))
for i, lesion_type in enumerate(lesion_type_map.keys()):
    # Get the filename of the first image of this type
    filename = metadata_df[metadata_df['dx'] == lesion_type]['image_id'].iloc[0] + '.jpg'
    image_path = find_image_path(filename)
    
    if image_path:
        # Read and display the image
        img = mpimg.imread(image_path)
        plt.subplot(3, 3, i + 1)  # Adjusted the grid size for readability
        plt.imshow(img)
        plt.title(lesion_type_map[lesion_type], fontsize=8)
        plt.axis('off')
    else:
        print(f"Image {filename} not found!")

plt.tight_layout()
plt.show()
