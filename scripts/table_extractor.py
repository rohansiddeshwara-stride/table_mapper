# from google.colab.patches import cv2_imshow
import fitz
import cv2
import numpy as np
import pandas as pd

from extractors import Borderless_Table_digital



def get_table(doc_path,page_no,bbox):
  doc = fitz.open(doc_path)
  zoom_x = 1.0
  zoom_y = 1.0
  if page_no < (len(doc)-1):
    page = doc[page_no]
    mat = fitz.Matrix(zoom_x, zoom_y)
    pix = page.get_pixmap(matrix=mat)
    pix.save("page.png")
    gen_table = Borderless_Table_digital(bbox, page_no, doc_path)
    table_json = gen_table.execute(page_no,1)
  return table_json

image=cv2.imread("page.png")