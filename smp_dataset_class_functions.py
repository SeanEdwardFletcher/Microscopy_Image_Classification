import cv2
import numpy as np
from scipy.stats import mode


def apply_mask(image, mask):
    """
    Applies a binary mask to the original image.

    Parameters:
    - image: The original grayscale image.
    - mask: A binary mask (same size as the image) where non-zero pixels are retained.

    Returns:
    - masked_image: The original image with masked regions set to white.
    """
    masked_image = image.copy()  # Create a copy of the original image
    masked_image[mask == 0] = 255  # Set masked (black) areas to white (255)
    return masked_image


def find_median_pixel_value(image):
    # Compute the median pixel value
    median_value = np.median(image)

    # print(f"Median Pixel Value: {median_value}")
    return median_value


def distance_from_central_value_of_image(image, num_std=1, central_tendency="median"):
    """
    Calculates how far the lower and upper bounds are from a chosen central tendency (mean, median, or mode)
    in terms of pixel intensity values.

    Parameters:
    - image_path: Path to the image file.
    - num_std: Number of standard deviations to include around the central value.
    - central_tendency: Method to determine the central value ('mean', 'median', or 'mode').

    Returns:
    - (distance_lower, distance_upper): How far the bounds are from the central tendency.
    """
    # Flatten image to 1D array of pixel values
    pixel_values = image.flatten()

    # Choose the central value based on user input
    if central_tendency == "mode":
        central_value = float(mode(pixel_values, keepdims=True).mode[0])
    elif central_tendency == "mean":
        central_value = np.mean(pixel_values)
    elif central_tendency == "median":
        central_value = np.median(pixel_values)
    else:
        raise ValueError("Invalid central_tendency. Choose 'mean', 'median', or 'mode'.")

    # Compute standard deviation of pixel values
    std_dev = np.std(pixel_values)

    # Compute how far the bounds are from the central value
    distance_from_center_value = min(num_std * std_dev, central_value)

    return distance_from_center_value


def inverse_band_pass(image, l_bound, u_bound):
    """Keeps only pixels outside the middle range (darker and brighter regions)."""
    _, low_mask = cv2.threshold(image, l_bound, 255, cv2.THRESH_BINARY_INV)
    _, high_mask = cv2.threshold(image, u_bound, 255, cv2.THRESH_BINARY)
    combined_mask = cv2.bitwise_or(low_mask, high_mask)
    return combined_mask


def inverse_bandpass_thresholding(the_image, st_d=0.5):
    blurred_image = cv2.GaussianBlur(the_image, (3, 3), 0)

    the_dist_frm_cntrl_val = distance_from_central_value_of_image(the_image, num_std=st_d)
    image_median = find_median_pixel_value(the_image)
    lower_bound = image_median - the_dist_frm_cntrl_val
    upper_bound = image_median + the_dist_frm_cntrl_val

    combined_mask = inverse_band_pass(blurred_image, lower_bound, upper_bound)
    masked_image = apply_mask(the_image, combined_mask)

    return masked_image



