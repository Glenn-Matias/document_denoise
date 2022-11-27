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
OUTPUT_EXTENSION = 'jpg'

TARGET = "target"
CREASE = "crease"
SHADOW = "shadow"
SHADOW_CREASE = 'shadow_and_crease'

# Change for varying inputs
NOISE_TYPE = CREASE

prefix = "output"
output_folder = f"{prefix}/ocr_preprocess"
corpus_name = "corpus.csv"
diffs_name = "diffs.csv"
percentages_name = "percentages.csv"
merged_name = 'merged.csv'
final_name = 'final.csv'

transformations_name = "transformations.csv"
DILATED = "inverted"
BLURRED = "blurred"
SUBTRACTED = 'subtracted'
INVERTED = "inverted"

BINARY = 'binary'
ADAPT = 'adapt'
OTSU = 'otsu'

def do_ocr(img):
    return pytesseract.image_to_string(img)

def create_output_folder(dir):
    Path(dir).mkdir(parents=True, exist_ok=True)


def create_preprocessed_corpus():

    create_output_folder(output_folder)
    df = pd.DataFrame(columns=['number', TARGET, CREASE, SHADOW, SHADOW_CREASE])


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
    df.to_csv(f"{output_folder}/{corpus_name}")

def create_preprocessed_diffs():
    corpus = pd.read_csv(f"{output_folder}/{corpus_name}")
    stats_df = pd.DataFrame(columns=['number', 'target_characters', 'crease_absolute_diff', 'shadow_absolute_diff', 'shadow_crease_absolute_diff'])

    for index, row in corpus.iterrows():
        print(f"Getting diffs @ {index}")
        target = row[TARGET]
        shadow = row[SHADOW]
        crease = row[CREASE]
        shadow_crease = row[SHADOW_CREASE]
        new_row = pd.DataFrame({'number': index+1,
                                'target_characters': len(row[TARGET]),
                                'crease_absolute_diff': edit_distance(target, crease),
                                'shadow_absolute_diff': edit_distance(target, shadow),
                                'shadow_crease_absolute_diff': edit_distance(target, shadow_crease)
                                }, index=[0])
        stats_df = stats_df.append(new_row, ignore_index=True)


    stats_df.to_csv(f"{output_folder}/{diffs_name}")
    print(stats_df)


def compute_preprocessed_relative_diffs():
    diffs = pd.read_csv(f"{output_folder}/{diffs_name}")
    stats_df = pd.DataFrame(columns=['number', 'crease_percentage_diff', 'shadow_percentage_diff', 'shadow_crease_percentage_diff'])

    for index, row in diffs.iterrows():
        print(f"Getting diffs @ {index}")

        target = int(row["target_characters"])
        shadow_perc = (int(row["crease_absolute_diff"]) / target) * 100
        crease_perc = (int(row["shadow_absolute_diff"]) / target) * 100
        shadow_crease_perc = (int(row["shadow_crease_absolute_diff"]) / target) * 100

        new_row = pd.DataFrame({'number': index+1,
                                        'crease_percentage_diff': shadow_perc,
                                        'shadow_percentage_diff': crease_perc,
                                        'shadow_crease_percentage_diff': shadow_crease_perc
                                        }, index=[0])
        stats_df = stats_df.append(new_row, ignore_index=True)

    stats_df.to_csv(f"{output_folder}/{percentages_name}")


def create_preprocessed_csv():
    corpus = pd.read_csv(f"{output_folder}/{corpus_name}")
    diffs_df = pd.read_csv(f"{output_folder}/{diffs_name}")
    percentages_df = pd.read_csv(f"{output_folder}/{percentages_name}")
    
    df = pd.merge(diffs_df, percentages_df, on='number')
    df = pd.merge(df, corpus[['number', 'target']], on='number')


    df.to_csv(f"{output_folder}/{merged_name}")
# One time things:
# create_preprocessed_corpus()
# create_preprocessed_diffs()
# compute_preprocessed_relative_diffs()


def create_postprocessed_corpus():
    output_folder = f"{prefix}/ocr_postprocess"

    transformations_list = [DILATED, BLURRED, SUBTRACTED, INVERTED, BINARY, ADAPT, OTSU]
    df = pd.DataFrame(columns=['number'] + transformations_list)

    for dataset_type in [CREASE, SHADOW, SHADOW_CREASE]:
        for image_number in range(1,11):
            row = [str(image_number)]
            for transformation in transformations_list:
                filename = f"output/{dataset_type}/{image_number}/{transformation}.{OUTPUT_EXTENSION}"
                img, width, height = read_image(filename)
                scanned_text = do_ocr(img)
            
                print(filename)

                row += [scanned_text]

            df.loc[image_number] = row
        df.to_csv(f"{output_folder}/{dataset_type}_{transformations_name}")

    # print(df)



def create_postprocessed_diffs():
    
    for dataset_type in [CREASE, SHADOW, SHADOW_CREASE]:
        if dataset_type != SHADOW_CREASE: continue
        print(f"*********{dataset_type}")
        merged_df = pd.read_csv(f"{output_folder}/{merged_name}")
        corpus = pd.read_csv(f"output/ocr_postprocess/{dataset_type}_{transformations_name}")

        df = pd.merge(merged_df, corpus, on='number')

        
        def func(target, transformed):
            return edit_distance(target, transformed)

        for transformation in [INVERTED, BINARY, ADAPT, OTSU]:
            # df[f'{transformation}_absolute_dif'] = edit_distance(target, crease)
            print(transformation)
            df[f'{transformation}_absolute_dif'] = df.apply(lambda row : func(row['target'], row[transformation]), axis = 1)
            df[f'{transformation}_percentage_dif'] = (df[f'{transformation}_absolute_dif'] / df['target_characters']) * 100

        df = df.filter(regex=(".percentage_dif"))
        df.to_csv(f"{prefix}/ocr_postprocess/{dataset_type}_{final_name}")

create_postprocessed_diffs()
# create_postprocessed_diffs()