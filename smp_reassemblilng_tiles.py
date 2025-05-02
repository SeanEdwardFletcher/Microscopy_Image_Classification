from PIL import Image
import os
import json

def apply_color_tint(image, tint_color):
    """Applies a color tint to a grayscale image."""
    if tint_color is None:
        return image

    if image.mode != 'RGB':
        image = image.convert('RGB')

    r, g, b = image.split()

    if tint_color.lower() == "blue":
        tinted = Image.merge('RGB', (r.point(lambda p: 0), g.point(lambda p: 0), b))
    elif tint_color.lower() == "green":
        tinted = Image.merge('RGB', (r.point(lambda p: 0), g, b.point(lambda p: 0)))
    elif tint_color.lower() == "red":
        tinted = Image.merge('RGB', (r, g.point(lambda p: 0), b.point(lambda p: 0)))
    elif tint_color.lower() == "purple":
        tinted = Image.merge('RGB', (r, g.point(lambda p: 0), b))
    elif tint_color.lower() == "yellow":
        tinted = Image.merge('RGB', (r, g, b.point(lambda p: 0)))
    elif tint_color.lower() == "cyan":
        tinted = Image.merge('RGB', (r.point(lambda p: 0), g, b))
    else:
        raise ValueError(f"Unsupported tint color: {tint_color}")

    return tinted

def reassemble_image(patches_dir, full_image_name, rows=6, cols=7, save_path=None,
                     border_thickness=0, border_color=(255, 0, 0), color_tint=None,
                     predictions_json=None, blend_red_blue=False):
    """
    Reassembles a full image from its patches, with optional coloring based on prediction correctness.
    """
    patches = []
    patch_predictions = {}

    if predictions_json:
        for pred in predictions_json["predictions"]:
            patch_predictions[pred["image"]] = {
                "ground_truth": pred["ground_truth"],
                "predicted": pred["predicted"],
                "probabilities": pred["probabilities"]
            }

    for i in range(1, rows * cols + 1):
        patch_filename = f"{full_image_name}_{i:02d}.jpg"
        patch_path = os.path.join(patches_dir, patch_filename)

        if not os.path.exists(patch_path):
            raise FileNotFoundError(f"Missing patch: {patch_path}")

        patch = Image.open(patch_path)
        if patch.mode != 'RGB':
            patch = patch.convert('RGB')

        pred_info = patch_predictions.get(patch_filename, None)

        if pred_info:
            patch_copy = patch.copy()
            pixels = patch_copy.load()
            width, height = patch_copy.size

            ground_truth = pred_info["ground_truth"]
            predicted = pred_info["predicted"]
            probabilities = pred_info["probabilities"]

            if blend_red_blue:
                # Blend red and blue according to probability strength
                blue_strength = probabilities[ground_truth]
                red_strength = sum(probabilities) - blue_strength

                total = blue_strength + red_strength
                if total > 0:
                    blue_strength /= total
                    red_strength /= total

                blended_color = (
                    int(255 * red_strength),  # Red
                    0,                        # Green
                    int(255 * blue_strength)   # Blue
                )

                for x in range(width):
                    for y in range(height):
                        original_pixel = pixels[x, y]
                        intensity = sum(original_pixel) / 3 / 255
                        new_pixel = tuple(int(intensity * c) for c in blended_color)
                        pixels[x, y] = new_pixel

                patch = patch_copy

            else:
                # Hard tint based on correctness
                correct = (predicted == ground_truth)
                tint = "blue" if correct else "red"
                patch = apply_color_tint(patch, tint)

        else:
            # No prediction available, fallback to optional tint
            patch = apply_color_tint(patch, color_tint)

        # Apply optional border
        if border_thickness > 0:
            patch = patch.copy()
            pixels = patch.load()
            width, height = patch.size
            for x in range(width):
                for y in range(height):
                    if (x < border_thickness or x >= width - border_thickness or
                        y < border_thickness or y >= height - border_thickness):
                        pixels[x, y] = border_color

        patches.append(patch)

    patch_width, patch_height = patches[0].size
    full_width = patch_width * cols
    full_height = patch_height * rows
    reassembled_image = Image.new('RGB', (full_width, full_height))

    for idx, patch in enumerate(patches):
        row = idx // cols
        col = idx % cols
        left = col * patch_width
        upper = row * patch_height
        reassembled_image.paste(patch, (left, upper))

    if save_path:
        reassembled_image.save(save_path)
        print(f"✅ Reassembled image saved to {save_path}")

    return reassembled_image


# === Example usage ===

# Load predictions
with open(r"C:\Users\fletc\PycharmProjects\Thesis\seans_project\test_results\04_18__0\test_results_knn_20250418_090336.json", "r") as f:
    predictions = json.load(f)

# Reassemble (NOT blended, just hard-tinted blue/red)
reassembled = reassemble_image(
    patches_dir=r"C:\Users\fletc\OneDrive\Desktop\split_taxol_ds_110_tiled_6x7\test\b_20nm",
    full_image_name="b_20nm_072",
    rows=6,
    cols=7,
    save_path=r"C:\Users\fletc\OneDrive\Desktop\MS_ComSci\Thesis\presentations\smp_presentation\reassembled_images\b_20nm_072_predblend_grid_.jpg",
    border_thickness=2,
    border_color=(255, 0, 0),
    predictions_json=predictions,
    blend_red_blue=True
)

reassembled.show()
