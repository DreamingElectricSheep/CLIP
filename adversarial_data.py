import cv2
import numpy as np
from PIL import Image

def add_gaussian_noise(image, sigma, grid_size=None, box_size=None, center_patch=None):
    """
    Sigma: 0-128
    grid_size: The number of patches per side (e.g., 14 for B/16)
    box_size: int (n), creates an n x n box of noisy patches.
    center_patch: tuple (row, col) defining the center of the box.
    """
    # Generate Gaussian noise
    mean = 0
    noise = np.random.normal(mean, sigma, image.shape)

    if box_size is not None and center_patch is not None:
        mask = np.zeros(image.shape, dtype=bool)
        h, w, _ = image.shape
        patch_h, patch_w = h // grid_size, w // grid_size
        
        c_row, c_col = center_patch
        # Calculate start/end offsets to keep the box centered
        # e.g., if n=3, radius is 1. (Row-1 to Row+1)
        radius = box_size // 2 
        
        # Calculate grid boundaries (clamped to prevent index errors)
        row_start = max(0, c_row - radius)
        row_end = min(grid_size, c_row + radius + (1 if box_size % 2 != 0 else 0))
        col_start = max(0, c_col - radius)
        col_end = min(grid_size, c_col + radius + (1 if box_size % 2 != 0 else 0))

        # Apply noise only to the pixels within the n x n patch range
        mask[row_start * patch_h : row_end * patch_h, 
             col_start * patch_w : col_end * patch_w, :] = True
            
        noise[~mask] = 0

    # Add the noise to the image
    noisy_image = np.clip(image.astype(float) + noise, 0, 255).astype('uint8')

    return np.clip(noisy_image, 0, 255)

def iterate_gaussian_noise(image, n: int, grid_size=None, box_size=None, center_patch=None) -> list:
    """
    Simulates real word corruption of data
    Takes in an image and returns a list of n noisy versions, OpenCV format.
    
    grid_size, box_size and center_patch are passed through to allow regional stress-testing.
    """
    if n <= 1:
        return [image] if isinstance(image, np.ndarray) else [cv2.imread(image)]

    if isinstance(image, str):
        image = cv2.imread(image)

    # Number of images returned
    output = []

    # Default is set to 128 as the maximum SD
    maximum = 128

    for i in range(n):
        if grid_size is not None and box_size is not None and center_patch is not None and i == 0:
            # Skip the first image
            continue
        current_sigma = maximum * i / (n - 1)
        temp = add_gaussian_noise(image, current_sigma, grid_size, box_size, center_patch)
        # print(current_sigma)
        # cv2.imshow('Gaussian Noise', temp)
        # cv2.waitKey(0)
        output.append(temp)
    
    # cv2.destroyAllWindows()
    return output
                      
def add_salt_and_pepper(image, prob):
    """
    Prob: 0-1
    """
    output = np.copy(image)
    # Salt (white pixels)
    num_salt = np.ceil(prob * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    output[tuple(coords)] = 255

    # Pepper (black pixels)
    num_pepper = np.ceil(prob * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    output[tuple(coords)] = 0
    return output

def iterate_salt_pepper(image, n: int) -> list:
    """
    Simulates dead pixels
    Takes in an image and returns a list of n distorted versions using salt and pepper, in PIL format
    """
    n = n - 1
    # Number of images returned
    output = []
    image = cv2.imread(image)

    for i in range(n + 1):
        temp = add_salt_and_pepper(image, i/n)
        color_coverted = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(color_coverted)
        output.append(pil_img)
    
    return output

def rotation(image) -> list:
    """
    Rotates an image 90 degrees 4 times
    """

    output = []
    image = cv2.imread(image)
    output.append(image)

    for i in range(3):
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        # cv2.imshow("Rotated:", image)
        # cv2.waitKey(0)

        output.append(image)

    return output

def brightness(image: str, n: int, amount: int) -> list:
    """
    Brightens and darkens an image n times by amount and returns an array containing these images
    n = 2 -> 5 outputs, with 2 bright, 2 dark, 1 original
    """

    output = []
    image = cv2.imread(image)
    output.append(image)
    temp = image

    for i in range(n):
        temp = cv2.add(temp, np.array([amount]))
        output.append(temp)
    
    temp = image
    for i in range(n):
        temp = cv2.subtract(temp, np.array([amount]))
        output.append(temp)

    # for i in output:
    #     cv2.imshow("Brightness: ", i)
    #     cv2.waitKey(0)
    return output

def pixelate(image, n) -> list:
    """
    Pixelates an image by double n times 
    """
    output = []
    image = cv2.imread(image)
    # output.append(image)
    if image is None:
        return []
    h, w = image.shape[:2]

    for i in range(1, n + 1):
        # Calculate divisor as 2^i for exponential "doubling" of pixel size
        divisor = 2**i 
        small_w, small_h = w // divisor, h // divisor
        
        # Stop if the image becomes too small
        if small_w <= 0 or small_h <= 0:
            break
            
        # Downsample
        temp = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        # Upsample back to original size (INTER_NEAREST preserves the blocks)
        temp = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        output.append(temp)
        # cv2.imshow("pixelate: ", temp)
        # cv2.waitKey(0)

    return output
