import cv2
import numpy as np

def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    return img

def resize_image(img):
    print('Original Dimensions : ',img.shape)
    
    scale_percent = 50 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    print('Resized Dimensions : ',resized.shape)
        
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

img = read_image("datasets/shadow_and_crease/10.jpeg")
# img = read_image("datasets/shadow/1.jpeg")
# img = resize_image(img)
original_image = img

img = dilate_image(img, kernel_size=8)
img = blur_image(img)
img = subtract_images(img, original_image)
img = invert_image(img)

before_and_after_img = np.hstack((original_image, img))
cv2.imshow('Image size {img.shape}', before_and_after_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

# from plotly import express as px
# px.imshow(dilated_img)