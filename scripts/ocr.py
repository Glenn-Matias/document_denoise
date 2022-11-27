import cv2
import numpy as np
import pytesseract
from PIL import Image
import pandas as pd 
from pathlib import Path
from nltk.metrics.distance import *


def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    width = img.shape[1]
    height = img.shape[0]
    return img, width, height

ROOT_DIR = 'datasets'
EXTENSION = 'jpeg'

TARGET = "target"
CREASE = "crease"
SHADOW = "shadow"
SHADOW_CREASE = 'shadow_and_crease'

# Change for varying inputs
NOISE_TYPE = CREASE

prefix = "output"
output_folder = f"{prefix}/ocr_preprocess"
    
def do_ocr(img):
    return pytesseract.image_to_string(img)

def create_output_folder(dir):
    Path(dir).mkdir(parents=True, exist_ok=True)


def create_preprocessed_ocr():

    create_output_folder(output_folder)
    df = pd.DataFrame(columns=['number', 'target', 'crease', 'shadow', 'shadow_crease'])


    for image_number in range(1,11):
        row = [str(image_number)]
        for dataset_type in [TARGET, CREASE, SHADOW, SHADOW_CREASE]:
            filename = f"{ROOT_DIR}/{dataset_type}/{image_number}.{EXTENSION}"
            img, width, height = read_image(filename)
            scanned_text = do_ocr(img)
        
            print(filename)

            row += [scanned_text]


        df.loc[image_number] = row

    print(df)
    df.to_csv(f"{output_folder}/all.csv")

# create_preprocessed_ocr()

df = pd.read_csv(f"{output_folder}/all.csv")


for index, row in df.iterrows():
    # print(row['target'], row['shadow'])
    first_text = row['target']
    second_text = row['shadow']

    
    print(edit_distance(first_text, first_text))
    break






