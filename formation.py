import os
import shutil
from sklearn.model_selection import train_test_split

# Function to count files in a folder
def count_files_in_folder(folder_path):
    return sum([len(files) for _, _, files in os.walk(folder_path)])

# Define source folder and destination folders
source_dir = 'brain_mri'
train_dir = 'brain_mri_train'
test_dir = 'brain_mri_test'

# Show the number of files in the original brain_mri folder
print("\nNumber of files in original brain_mri folder:")

# Initialize variable to track the total number of files in the original dataset
original_files_count = 0

# Count files in the original folder
for class_folder in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_folder)
    if os.path.isdir(class_path):
        class_files = [img for img in os.listdir(class_path) if img.endswith('.jpg')]
        print(f"{class_folder}: {len(class_files)} files")
        original_files_count += len(class_files)

print(f"Total files in original brain_mri folder: {original_files_count} files")

# Create directories for train and test if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Iterate over each class folder to perform the split and move files
for class_folder in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_folder)

    if os.path.isdir(class_path):
        # Create class-specific folders in the train and test directories
        train_class_path = os.path.join(train_dir, class_folder)
        test_class_path = os.path.join(test_dir, class_folder)
        
        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(test_class_path, exist_ok=True)

        # Get all the images in the current class folder
        images = [img for img in os.listdir(class_path) if img.endswith('.jpg')]

        # Split images into train and test sets
        train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)

        # Move the files to their respective train/test folders (moving files into new directories)
        for img in train_images:
            shutil.move(os.path.join(class_path, img), os.path.join(train_class_path, img))

        for img in test_images:
            shutil.move(os.path.join(class_path, img), os.path.join(test_class_path, img))

# After the split, count the total number of files in train and test directories
train_files_count = count_files_in_folder(train_dir)
test_files_count = count_files_in_folder(test_dir)

# Print the number of files in the train and test directories
print("\nNumber of files in each folder:")
print(f"Total files in train folder: {train_files_count}")
print(f"Total files in test folder: {test_files_count}")

# Verify that the total number of files in train and test matches the original count
assert train_files_count + test_files_count == original_files_count, (
    f"Error: The total number of files in train and test ({train_files_count + test_files_count}) "
    f"does not match the original count ({original_files_count})."
)

print("\nTrain and Test sets have been created, and the file counts match.")
