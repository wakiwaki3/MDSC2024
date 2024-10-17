import sys
import pytesseract
from PIL import Image
from pathlib import Path
import glob
import pandas as pd
import cv2
import numpy as np

def pil2cv(image):
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        new_image = cv2.cvtColor(new_image, cv2.COLOR_GRAY2BGRA)
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def cv2pil(image):
    new_image = image.copy()
    if new_image.ndim == 2:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_GRAY2RGB)
    elif new_image.shape[2] == 3:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

def preprocessing_image(img):
    cvimg = pil2cv(img)
    # gray_image = cv2.cvtColor(cvimg, cv2.COLOR_BGR2GRAY)
    # denoised_image = cv2.bilateralFilter(gray_image, 9, 75, 75)
    # cvimg = cv2.cvtColor(denoised_image, cv2.COLOR_GRAY2BGR)
    # mimg = morf(im_edges)
    return cv2pil(cvimg)#gray_image)

def trans_answer_s(s):
    dic = {"．":".", "，":",", "？":"?", "！":"!", "：":":", "；":";", "／":"/", "＜":"<", "＞":">", "＝":"=", "＋":"+", "ー":"-", "＆":"&", "％":"%", "＃":"#", "￥":"\\", "（":"(", "）":")", "［":"[", "］":"]"}
    trans = str.maketrans(dic) 
    return s.translate(trans)

def image_to_text(image_path):
    img = Image.open(image_path)
    img = preprocessing_image(img)

    text = pytesseract.image_to_string(img, lang='jpn')
    text = text.replace(" ", "")
    text = trans_answer_s(text)
    textlist = text.split("\n")

    return textlist

def get_filenames(img_dir_path):
    res = glob.glob(str(Path(img_dir_path) / "*.png"))
    return res

def solver(image_dir_path):
    output_list = []
    output_file_name = "オープン‗FT‗脇上和也.tsv"
    filenames = get_filenames(image_dir_path)
    for i, filename in enumerate(filenames):
        result_list = []
        result_list.append(Path(filename).name)
        textlist = image_to_text(filename)
        for text in textlist:
            result_list.append(text)
        output_list.append(result_list)
    df = pd.DataFrame(output_list)
    df.to_csv(output_file_name, sep="\t", index=False, header=False, encoding="utf_8", lineterminator='\n')

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_dir_path = sys.argv[1]  # コマンドライン引数から画像ファイルのパスを取得
        solver(image_dir_path)
    else:
        print("Please input dirpath with argv")