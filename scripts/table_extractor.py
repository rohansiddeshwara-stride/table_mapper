# from google.colab.patches import cv2_imshow
import fitz
import cv2
import numpy as np
import pandas as pd

from extractors import Borderless_Table_digital



def get_table(doc_path,page_no,bbox):
  doc = fitz.open(doc_path)
  zoom_x = 3.0
  zoom_y = 3.0
  if page_no < (len(doc)-1):
    page = doc[page_no]
    # mat = fitz.Matrix(zoom_x, zoom_y)
    # pix = page.get_pixmap(matrix=mat)
    # pix.save("page.png")
    gen_table = Borderless_Table_digital(bbox, page_no, doc_path)
    table_json = gen_table.execute(page_no,1)

    # image=cv2.imread("page.png")

    # cv2.rectangle(image, (round(bbox[0]*zoom_x), round(bbox[1]*zoom_y)),  (round(bbox[2]*zoom_x), round(bbox[3]*zoom_y)), (20,200,0), 2)
    # for d in table_json["data"]:
    #   for cell in d['row_value']:
    #     c=cell
    #     cv2.rectangle(image, (round(c["bbox"][0]*zoom_x), round(c["bbox"][1]*zoom_y)),  (round(c["bbox"][2]*zoom_x), round(c["bbox"][3]*zoom_y)), (0,0,255), 2)
    # cv2.imwrite("page.png",image)
  return table_json


