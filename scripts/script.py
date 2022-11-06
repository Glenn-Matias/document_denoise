import cv2
import numpy as np

def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    return img

def resize_image(img, scale=0.5):    
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    resized = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
    return resized


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
NOISE_TYPE = SHADOW_CREASE

# for image_number in range(1,11):
for image_number in range(1,11):
    # if image_number!=10: continue
    filename = f"D:\Documents\DLSU\Year 4 Term 1\CV\document_denoise\{ROOT_DIR}\{NOISE_TYPE}\{image_number}.{EXTENSION}"

    img = read_image(filename)
    original_image = img
    # cv2.imshow("image", img)

    img = dilate_image(img, kernel_size=DILATE_KERNEL_SIZE)
    # cv2.imshow("image", img)
    # img = blur_image(img)
    img = subtract_images(img, original_image)
    # cv2.imshow("image", img)
    img = invert_image(img)
    # cv2.imshow("image", img)

    ret, binary_img = cv2.threshold(img,170,255, cv2.THRESH_BINARY) 
    cv2.imshow("image", binary_img)
    ret, otsu_img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU) # 2nd param doesnt matter
    # cv2.imshow("image", otsu_img)
    # adapt_img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,101,40)
    adapt_img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,101,40)
    # cv2.imshow("image", adapt_img)

    # Resize for viewport
    original_image = resize_image(original_image, scale=IMAGE_SCALE)
    img = resize_image(img, scale=IMAGE_SCALE)
    binary_img = resize_image(binary_img, scale=IMAGE_SCALE)
    adapt_img = resize_image(adapt_img, scale=IMAGE_SCALE)
    otsu_img = resize_image(otsu_img, scale=IMAGE_SCALE)

    before_and_after_img = np.hstack((original_image, binary_img, adapt_img, otsu_img))
    # cv2.imshow(f'Image size {filename}', before_and_after_img)


cv2.waitKey(0)
cv2.destroyAllWindows()

# from plotly import express as px
# px.imshow(dilated_img)

