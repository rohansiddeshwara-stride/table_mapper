import os
from flask import Flask, request, jsonify
import json

from scripts import get_table



app = Flask(__name__)

# Function that you want to call
def process_pdf(pdf_path, page_no, bbox):
    
    result = get_table(pdf_path,int(page_no),list(bbox))
    return result

@app.route('/', methods=['POST'])
def api_process_pdf():
    data = request.get_json()

    if 'pdf_path' not in data or 'page_no' not in data or 'bbox' not in data:
        return jsonify({"error": "Missing one or more required fields."}), 400

    pdf_path = data['pdf_path']
    page_no = data['page_no']
    bbox = data['bbox']

    if not isinstance(page_no, int) or page_no <= 0:
        return jsonify({"error": "page_no should be a positive integer."}), 400

    if not isinstance(bbox, list) or len(bbox) != 4:
        return jsonify({"error": "bbox should be a list of 4 integers."}), 400

    result = process_pdf(pdf_path, page_no, bbox)
    return jsonify(result)

if __name__ == '__main__':

    doc_path='/workspaces/table_mapper/BMW 2009-1 Vehicle Lease Trust - Aug 09.pdf'
    page_no=0
    bbox=[50,111,400,200]
    print(process_pdf(doc_path, page_no, bbox))
    # app.run(debug=True)
