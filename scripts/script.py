import cv2
import numpy as np

def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    width = img.shape[1]
    height = img.shape[0]
    return img, width, height

def resize_image(img, scale=0.5):    
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    resized = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
    return resized, width, height


def dilate_image(img, kernel_size=10):
    # Dilate the image: using the kernel, set the pixel at the middle based on the max value of what's overlapping the kernel. 
    # Setting kernel to large values made the text bled.
    dilated_img = cv2.dilate(img, np.ones((kernel_size,kernel_size), np.uint8))

    return dilated_img


def blur_image(img, kernel_size=21):
    blurred_image = cv2.medianBlur(img, kernel_size)

    return blurred_image

def subtract_images(img1, img2):
    difference_img = cv2.absdiff(img1, img2)
    return difference_img

def invert_image(img):
    inverted_img = 255 - img

    return inverted_img

def add_text_to_image(img, text):
    img = cv2.putText(img, text, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 3, cv2.LINE_AA)
    return img

def save_image(img, folder, filename):

    from pathlib import Path


    # img = add_text_to_image(img, filename)

    # img = cv2.resize(img, (600, 800))
    prefix = "output"
    dir = f"{prefix}/{folder}"
    filename = f"{dir}/{filename}.jpg"

    Path(dir).mkdir(parents=True, exist_ok=True)

    status = cv2.imwrite(filename, img)

    print(f'Saving to {filename}: {status}')



# Declare constants
DILATE_KERNEL_SIZE = 21
IMAGE_SCALE = 0.5
ROOT_DIR = 'datasets'
EXTENSION = 'jpeg'

GLARE = "glare"
CREASE = "crease"
SHADOW = "shadow"
SHADOW_CREASE = 'shadow_and_crease'

# Change for varying inputs
NOISE_TYPE = CREASE

# for image_number in range(1,11):
for image_number in range(1,11):
    if image_number!=1: continue
    filename = f"{ROOT_DIR}/{NOISE_TYPE}/{image_number}.{EXTENSION}"

    img, width, height = read_image(filename)
    original_image = img

    dilated_img = dilate_image(img, kernel_size=DILATE_KERNEL_SIZE)
    blurred_img = blur_image(dilated_img)
    subtracted_img = subtract_images(blurred_img, original_image)
    inverted_img = invert_image(subtracted_img)

    img = inverted_img


    ret, binary_img = cv2.threshold(img,170,255, cv2.THRESH_BINARY) 
    ret, otsu_img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU) # 2nd param doesnt matter
    # adapt_img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,101,40)
    adapt_img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,101,40)


    save_image(dilated_img, f"{NOISE_TYPE}/{image_number}", "dilated")
    save_image(blurred_img, f"{NOISE_TYPE}/{image_number}", "blurred")
    save_image(subtracted_img, f"{NOISE_TYPE}/{image_number}", "subtracted")
    save_image(inverted_img, f"{NOISE_TYPE}/{image_number}", "inverted")
    save_image(binary_img, f"{NOISE_TYPE}/{image_number}", "binary")
    save_image(otsu_img, f"{NOISE_TYPE}/{image_number}", "otsu")
    save_image(adapt_img, f"{NOISE_TYPE}/{image_number}", "adapt")

    # Resize for viewport
    original_image, scaled_width, scaled_height = resize_image(original_image, scale=IMAGE_SCALE)
    original_image = add_text_to_image(original_image, "Original")

    dilated_img, _, _ = resize_image(dilated_img, scale=IMAGE_SCALE)
    dilated_img = add_text_to_image(dilated_img, "Dilated")

    blurred_img, _, _ = resize_image(blurred_img, scale=IMAGE_SCALE)
    blurred_img = add_text_to_image(blurred_img, "Blurred")

    subtracted_img, _, _ = resize_image(subtracted_img, scale=IMAGE_SCALE)
    subtracted_img = add_text_to_image(subtracted_img, "Subtracted")

    inverted_img, _, _ = resize_image(inverted_img, scale=IMAGE_SCALE)
    inverted_img = add_text_to_image(inverted_img, "Inverted")

    before_and_after_img = np.hstack((original_image, dilated_img, blurred_img, subtracted_img, inverted_img))
    cv2.imshow(f'Steps before Thresholding {filename}', before_and_after_img)

    binary_img, _, _ = resize_image(binary_img, scale=IMAGE_SCALE)
    binary_img = add_text_to_image(binary_img, "Binary Threshold")

    adapt_img, _, _ = resize_image(adapt_img, scale=IMAGE_SCALE)
    adapt_img = add_text_to_image(adapt_img, "Adaptive Threshold")

    otsu_img, _, _ = resize_image(otsu_img, scale=IMAGE_SCALE)
    otsu_img = add_text_to_image(otsu_img, "Otsu's Threshold")

    before_and_after_img = np.hstack((original_image, binary_img, adapt_img, otsu_img))
    cv2.imshow(f'After Thresholding {filename}', before_and_after_img)

    


cv2.waitKey(0)
cv2.destroyAllWindows()

# from plotly import express as px
# px.imshow(dilated_img)

