from PIL import Image
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--destination_dir", required=False)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--rotate", action="store_true", help="Rotate images 90 degrees clockwise before tiling")

    return parser.parse_args()


def create_subimages(image, image_name, rows=2, cols=2):
    """
    Splits an image into subimages based on specified rows and columns, ignoring
    any remainder pixels that do not fit evenly into the grid.

    Parameters:
    - image (PIL.Image.Image): The input image to be divided.
    - image_name (str): The base name of the original image.
    - rows (int): Number of horizontal divisions (default is 2).
    - cols (int): Number of vertical divisions (default is 2).

    Returns:
    - subimages (list): A list of tuples in the form (subimage, subimage_name).
    """
    try:
        # Get image dimensions
        width, height = image.size

        # Calculate the width and height of each subimage
        sub_width = width // cols
        sub_height = height // rows

        # Prepare list to store subimages
        subimages = []
        sub_im_no = 1

        # Create subimages, ignoring remaining pixels
        for i in range(rows):
            for j in range(cols):
                # Define crop boundaries
                left = j * sub_width
                upper = i * sub_height
                right = left + sub_width
                lower = upper + sub_height

                # Crop the subimage (ignore remainder)
                subimage = image.crop((left, upper, right, lower))

                # Generate subimage name
                subimage_name = f"{image_name}_{sub_im_no:02d}"
                subimages.append((subimage, subimage_name))

                sub_im_no += 1

        return subimages

    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return []


def process_image(image_path, output_dir, base_name, rows=2, cols=2, rotate=False):
    """
    Processes a single image by splitting it into subimages and saving them.

    Parameters:
    - image_path (str): Path to the image to be processed.
    - output_dir (str): Directory where the subimages will be saved.
    - base_name (str): Base name to be used for subimages.
    - rows (int): Number of rows for subimage division.
    - cols (int): Number of columns for subimage division.
    """
    try:
        # Open the image
        image = Image.open(image_path)
        if rotate:
            image = image.transpose(Image.Transpose.ROTATE_270)  # 90 degrees clockwise
    except Exception as e:
        print(f"⚠️ Skipping file '{image_path}': {e}")
        return

    # Create subimages
    subimages = create_subimages(image, base_name, rows, cols)

    # Save each subimage to the destination directory
    for subimage, subimage_name in subimages:
        subimage_path = os.path.join(output_dir, f"{subimage_name}.jpg")
        subimage.save(subimage_path, "JPEG")

    num_subimages = len(subimages)
    print(f"✅ Processed '{base_name}' → {num_subimages} subimages saved to '{output_dir}'")


def process_directory(source_dir, destination_dir, rows=2, cols=2, rotate=False):
    """
    Traverses the source directory and processes all .tif, .jpg, and .jpeg files.

    Parameters:
    - source_dir (str): Path to the source directory.
    - destination_dir (str): Path to the destination directory.
    - rows (int): Number of rows for subimage division.
    - cols (int): Number of columns for subimage division.
    """
    total_images = 0  # Track total images processed

    for root, dirs, files in os.walk(source_dir):
        # Skip processing files inside the destination directory
        if destination_dir in root:
            continue

        # Filter only .tif, .jpg, and .jpeg files in the current source directory
        image_files = [file for file in files if file.lower().endswith((".tif", ".jpg", ".jpeg"))]

        for file in image_files:
            image_path = os.path.join(root, file)

            # Create corresponding subdirectory in the destination directory
            relative_path = os.path.relpath(root, source_dir)
            output_dir = os.path.join(destination_dir, relative_path)
            os.makedirs(output_dir, exist_ok=True)

            # Get original filename without extension
            base_name = os.path.splitext(file)[0]

            # Process the individual image
            process_image(image_path, output_dir, base_name, rows, cols, rotate)

            total_images += 1

    print(f"\n🎉 Processing complete! {total_images} images processed and split into {rows * cols} sub-images each.")


def process_images_in_directory(source_dir, destination_dir, rows=2, cols=2, rotate=False):
    """
    Main umbrella function to handle directory creation and call processing.

    Parameters:
    - source_dir (str): Path to the source directory.
    - destination_dir (str): Path to the destination directory.
    - rows (int): Number of rows for subimage division.
    - cols (int): Number of columns for subimage division.
    """
    # Ensure the destination directory exists
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Start processing the source directory
    process_directory(source_dir, destination_dir, rows, cols, rotate)


def run_subimage_script(source_dir, target_dir=None, rows=2, cols=2, rotate=False):
    if target_dir:
        destination_directory = target_dir
    else:
        # Automatically name destination folder based on tiling dimensions
        parent_dir = os.path.dirname(source_dir.rstrip("/\\"))
        base_name = os.path.basename(source_dir.rstrip("/\\"))
        destination_directory = os.path.join(parent_dir, f"{base_name}_tiled_{rows}by{cols}")

    process_images_in_directory(source_dir, destination_directory, rows=rows, cols=cols, rotate=rotate)


# Run the script when executed directly
if __name__ == "__main__":
    args = parse_args()
    run_subimage_script(
        source_dir=args.image_dir,
        target_dir=args.destination_dir,
        rows=args.rows,
        cols=args.columns,
        rotate=args.rotate
    )

# exampl call
r"""
python smp_create_sub_images.py --image_dir "C:\Users\fletc\OneDrive\Desktop\split_taxol_ds_110" --destination_dir "C:\Users\fletc\OneDrive\Desktop\split_taxol_ds_110_tiled_6x7" --rows 6 --columns 7

A
"""
