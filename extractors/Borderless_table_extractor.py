import fitz
import cv2
import numpy as np
import pandas as pd

class Borderless_Table_digital():
  def __init__(self,table_bbox,page_no,document):
    self.doc = fitz.open(document)
    self.page = self.doc.load_page(page_no)
    self.text_instances = self.page.get_text("words")
    # for word in self.text_instances:
    #   print(word)
    # print(self.text_instances)
    self.table_rect=table_bbox
    self.bbox=table_bbox
    self.selected_text=[]

  def execute(self,page_no,table_no):
    self.extract_words()
    self.bounding_boxes = self.combine_words_along_x(distance_threshold=5)
    self.bounding_boxes = self.combine_dollar(self.bounding_boxes)
    self.all_cells=self.bounding_boxes.copy()
    if len(self.bounding_boxes)< 1:self.bounding_boxes=[[0,0,1,1," "]]
    self.mean_height =self.get_mean_height_of_bounding_boxes()
    self.mean_width=self.get_mean_width_of_bounding_boxes()


    self.bounding_boxes = self.sort_bounding_boxes_by_y_coordinate(self.bounding_boxes)
    # print(bounding_boxes)
    self.rows=[]
    self.rows = self.club_all_bounding_boxes_by_similar_y_coordinates_into_rows(self.mean_height,self.bounding_boxes,self.rows)
    self.rows = self.sort_all_rows_by_x_coordinate(self.rows)


    self.no_columns=0
    self.no_columns = self.get_no_of_columns(self.rows)
    column_range = self.get_column_range(self.rows,self.no_columns)
    self.columns =self.get_columns(self.all_cells,column_range)

    self.table_with_cell_bbox, self.table_without_cell_bbox = self.create_final_table(self.rows,self.columns)


    # self.generate_csv_file(self.table_without_cell_bbox,page_no,table_no)

    table_extracted=self.generate_json(self.table_with_cell_bbox,page_no,self.bbox)
    return table_extracted


  def intersects_rect(self,table_rect,bbox):
    if table_rect[0]<=bbox[0] and table_rect[1]<=bbox[1] and table_rect[2]>=bbox[2] and table_rect[3]>=bbox[3]:
      return True
    else:
      return False

  def extract_words(self):
    for inst in self.text_instances:
      bbox = inst[0:4]
      if self.intersects_rect(self.table_rect,bbox):
          self.selected_text.append(inst[0:5])


  def combine_words_along_x(self, distance_threshold=10):
    bbox_list = self.selected_text
    combined_words = []
    amounts_combined =[]


    i = 0
    while i < len(bbox_list):
        curr_bbox = bbox_list[i]
        combined_text = curr_bbox[4]
        x1, y1, x2, y2 = curr_bbox[:4]

        j = i + 1
        while j < len(bbox_list):
            next_bbox = bbox_list[j]
            next_x1, next_y1, _, _ = next_bbox[:4]

            if (abs(next_x1 - x2 )<= distance_threshold and abs(next_y1 - y1 )<1) :
            #  (combined_text!='$' or next_bbox[4] !='$'):
                # print('1',combined_text)
                combined_text += " " + next_bbox[4]
                x2 = next_bbox[2]
                j += 1
                # print(next_bbox[4])
            else:
                # if combined_text =="$":
                #   # print('2',combined_text)
                #   next_word = self.get_closest_word(curr_bbox,bbox_list)
                #   amounts_combined.append(next_word)
                #   combined_text += " " + next_word[4]
                #   x2 = next_word[2]

                # else:
                #   # print('3',combined_text)
                  break

        combined_words.append((x1, y1, x2, y2, combined_text))
        i = j

    # for ce in amounts_combined:
    #   if ce in combined_words:

    #     combined_words.remove(ce)

    return combined_words

  def combine_dollar(self,all_bboxes):
    all_bboxes.sort(key=lambda x:x[3])
    filtred_bbox = []
    t =[]
    for bbox in all_bboxes:
      if bbox[4] == '$':
          next_word = self.get_closest_word(bbox,all_bboxes)
          new_cell = [min(bbox[0],next_word[0]),min(bbox[1],next_word[1]),max(bbox[2],next_word[2]),max(bbox[3],next_word[3]),bbox[4]+" "+next_word[4]]
          filtred_bbox.append(new_cell)
          # print(new_cell[4])
          t.append(next_word)
          t.append(bbox)

    for bbox in all_bboxes:
      if bbox not in t:
        filtred_bbox.append(bbox)


    return filtred_bbox

  def get_closest_word(self,current_word,words):
    distances =[]
    filtred_words =[]

    words.sort(key = lambda x : x[1])

    for word in words:
      if abs(current_word[1] - word[1]) < 2 and current_word != word and  current_word[2] < word[0] and  abs(current_word[2] - word[0]) < 100 :
        filtred_words.append(word)

    filtred_words.sort(key= lambda x:x[0])

    for w in filtred_words:
      distances.append(abs(current_word[0]-w[0]))

    index = np.argmin(distances)

    return filtred_words[index]


  def get_mean_height_of_bounding_boxes(self):
    heights = []
    for bounding_box in self.bounding_boxes:
        # print(bounding_box)
        x1, y1,x2 , y2,_ = bounding_box
        heights.append(abs(y2-y1))
    return np.mean(heights)

  def get_mean_width_of_bounding_boxes(self):
    widths = []
    for bounding_box in self.bounding_boxes:
        x1, y1, x2, y2 ,_ = bounding_box
        widths.append(abs(x2-x1))
    return np.mean(widths)

  def sort_bounding_boxes_by_y_coordinate(self,bounding_boxes):
      self.bounding_boxes = sorted(bounding_boxes, key=lambda x: x[1])
      # print(bounding_boxes)
      return self.bounding_boxes
  def sort_bounding_boxes_by_x_coordinate(self,bounding_boxes):
      self.bounding_boxes = sorted(bounding_boxes, key=lambda x: x[0])
      # print(bounding_boxes)
      return self.bounding_boxes

  def club_all_bounding_boxes_by_similar_y_coordinates_into_rows(self,mean_height,bounding_boxes,rows):
      half_of_mean_height = mean_height/3
      current_row = [ self.bounding_boxes[0] ]
      for bounding_box in bounding_boxes[1:]:
          current_bounding_box_y = bounding_box[1]
          previous_bounding_box_y = current_row[-1][1]
          distance_between_bounding_boxes = abs(current_bounding_box_y - previous_bounding_box_y)
          if distance_between_bounding_boxes <= 3:
              current_row.append(bounding_box)
          else:
              rows.append(current_row)
              current_row = [ bounding_box ]
      if not current_row[0]==" ":

        rows.append(current_row)

      # for r in rows:
      #   print(r)

      return rows



  def sort_all_rows_by_x_coordinate(self,rows):
      for row in rows:
          row.sort(key=lambda x: x[0])
      return rows


  def sort_all_columns_by_y_coordinate(self,columns):
      for column in columns:
          column.sort(key=lambda x: x[1])
      return columns



  def get_no_of_columns(self,rows):
    no_rows= len(rows)
    count_columns=[]
    for each_row in rows:
      count_columns.append(len(each_row))

    no_columns = max(count_columns)
    return no_columns


  # def get_column_range(self,rows,no_columns):

  #   # filter rows (all rows wiht no of columns are stored into temp_rows)
  #   temp_rows=[]
  #   for each_row in rows:
  #     if len(each_row)==no_columns:
  #         temp_rows.append(each_row)


  #   # create the range of all columns range->(x0,x1)
  #   # where x0 and x1 are the position of cell with max length
  #   column_range=[]
  #   for i in range(no_columns):
  #     l= [{"bbox":x[i][0:4], "len":abs(x[i][2]-x[i][0])} for x in temp_rows]

  #     max_length_bbox = max(l, key=lambda x: x.get('len', 0))['bbox']

  #     column_range.append([max_length_bbox[0],max_length_bbox[2]])

  #   # combine the cell ranges to get distinct start and end positions of the columns
  #   new_col_positions=[]
  #   table_bbox=self.bbox

  #   minium =  table_bbox[0]  if table_bbox[0]< column_range[0][0] else column_range[0][0]
  #   new_col_positions.append(minium)

  #   for  i in range(len(column_range)-1):
  #     avg = (column_range[i + 1][0] - column_range[i][1])/2
  #     new_range = column_range[i][1] + avg
  #     new_col_positions.append(new_range)

  #   # print(table_bbox[2],column_range[-1][1])
  #   new_col_positions.append(max(table_bbox[2],column_range[-1][1])+10)
  #   # new_col_positions.append(column_range[-1][1])
  #   new_col_range = [[new_col_positions[i], new_col_positions[i + 1]] for i in range(len(new_col_positions) - 1)]
  #   # for  i ,column in enumerate(column_range[:-1]):
  #   #   avg = (column_range[i + 1][0] - column_range[i][1])/2
  #   # new_col_range.append([new_col_positions[-1][1],table_bbox[3]])
  #   # print(new_col_range)
  #   return new_col_range

  def get_column_range(self,rows,no_columns):

    # filter rows (all rows wiht no of columns are stored into temp_rows)
    temp_rows=[]
    for each_row in rows:
      if len(each_row)==no_columns:
          temp_rows.append(each_row)


    # create the range of all columns range->(x0,x1)
    # where x0 and x1 are the position of cell with max length
    column_range=[]
    for i in range(no_columns):
      # l= [{"bbox":x[i][0:4], "len":abs(x[i][2]-x[i][0])} for x in temp_rows]
      # min_x0 = 1000
      # max_x1 = -1000
      # for x in temp_rows:
      #   if x[i][0] < min_x0:
      #     min_x0 = x[i][0]
      #   if x[i][2] > max_x1:
      #     max_x1 = x[i][2]
      x0_list = [x[i][0] for x in temp_rows]
      x1_list = [x[i][2] for x in temp_rows]
      min_x0 = min(x0_list)
      max_x1 = max(x1_list)



      column_range.append([min_x0,max_x1])

    # combine the cell ranges to get distinct start and end positions of the columns
    new_col_positions=[]
    table_bbox=self.bbox

    minium =  table_bbox[0]  if table_bbox[0]< column_range[0][0] else column_range[0][0]
    new_col_positions.append(minium)

    for  i in range(len(column_range)-1):
      avg = abs(column_range[i + 1][0] - column_range[i][1])/2
      new_range = column_range[i][1] + avg
      new_col_positions.append(new_range)

    new_col_positions.append(max(table_bbox[2],column_range[-1][1])+10)
    new_col_range = [[new_col_positions[i], new_col_positions[i + 1]] for i in range(len(new_col_positions) - 1)]
    return new_col_range

  def find_merge_col_range(self,cell, new_col_range):

    merge_cols = []

    for i, c in enumerate(new_col_range):
      if cell[0] >= c[0] and cell[2] >= c[1]:

        for j in range(i + 1, len(new_col_range)):

          if cell[0] <= new_col_range[j][0] and  cell[2] <= new_col_range[j][1]:
            merge_cols.append(j)
            merge_cols.append(i)
            # print("2", merge_cols)

            return merge_cols

          elif cell[0] <= new_col_range[j][0] and  cell[2] >= new_col_range[j][1]:
            merge_cols.append(j)

          else:
            break
            # print("3", merge_cols)
    return merge_cols

  def get_columns(self,all_cells,new_col_range):
    columns=[]
    merged_cells=all_cells.copy()
    # print(new_col_range)
    for c in new_col_range:
      column_arrage=[]
      for cell in all_cells:
        if cell[0]>=c[0] and cell[2] <= c[1]:
          column_arrage.append(cell)
          merged_cells=[sublist for sublist in merged_cells if sublist != cell]
      # print([c[4] for c in column_arrage])
      columns.append(column_arrage)
    for c in merged_cells:
      merge_cols = self.find_merge_col_range(c, new_col_range)
      #  print("eeeeeeeeee", merge_cols)
      for mc in merge_cols:
        columns[mc].append(c)
    # for c in columns:
    #   for x in c:
    #     print(x[4])
    #   print("////////////////////////////////////")
    return columns


  # def find_element_position(self,matrix, target_element):
  #   for row_index, row in enumerate(matrix):
  #       for col_index, element in enumerate(row):
  #           if element == target_element:
  #               matrix[row_index][col_index]=" "
  #               return row_index, col_index
  #   return None


  def generate_csv_file(self,table,page_no,table_no):
      # print(table)
      df=pd.DataFrame(table)
      df.to_csv(f"page{page_no}_table{table_no}.csv")


  def generate_json(self,table,page_no,bbox):
     table_dict={"page_no":page_no,
                 "bbox":[bbox],
                 "data":table}
     return table_dict


  def find_element_position(self, matrix, target_element):

    for row_index, row in enumerate(matrix):
        for col_index, element in enumerate(row):
          if element == target_element:
            # if flag==1:
            matrix[row_index][col_index]=" "
            return row_index, col_index, matrix
    return -1,-1, matrix


  def create_final_table(self,row_table,column_table):

    text=[t for sublist in column_table for t in sublist]

    row_len=len(row_table)
    col_len=len(column_table)

    table = [[" " for _ in range(col_len)] for _ in range(row_len)]

    self.table_with_bbox=[[{"cell_data":" ","cell_bbox":[-1,-1,-1,-1]} for _ in range(col_len)] for _ in range(row_len)]
    for i in text:
      row, _, row_table  = self.find_element_position(row_table,i)
      col, _, column_table = self.find_element_position(column_table,i)

      if (row>=0 and col>=0):
        table[row][col]=table[row][col]+" "+i[4]

        self.table_with_bbox[row][col]={"cell_data":i[4],"cell_bbox":[int(x) for x in i[0:4]]}

    return self.table_with_bbox, table

  def generate_json(self,table,page_no,bbox):
    table_data=[]
    for row_id ,row in enumerate(table):
      row_json ={"row_id":row_id,"row_value":[]}
      for cell_id,cell in enumerate(row):
        cell_json = {"bbox" : cell["cell_bbox"], "cell_id":str(row_id)+"_"+str(cell_id) ,"cell_value":cell["cell_data"]}
        row_json["row_value"].append(cell_json)
      table_data.append(row_json)

    table_dict={"page_no":page_no,
                 "bbox":[bbox],
                 "data":table_data}
    return table_dict
