import io

import torch
import json
import re
import tempfile
from pathlib import Path
import cv2
import numpy as np
import minify_html
import pandas as pd
from PIL import Image
from img2table.ocr import EasyOCR
from tqdm import tqdm, trange
from torchvision import transforms
from torchvision.ops import box_iou
from transformers import TableTransformerForObjectDetection
from transformers import AutoModelForObjectDetection
import easyocr
from img2table.document import Image as ImageT
from wakepy.modes import keep


# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Table%20Transformer/Inference_with_Table_Transformer_(TATR)_for_parsing_tables.ipynb
class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale * width)), int(round(scale * height))))

        return resized_image


def objects_to_crops(img, tokens, objects, class_thresholds, padding=10):
    """
    Process the bounding boxes produced by the table detection model into
    cropped table images and cropped tokens.
    """

    table_crops = []
    for obj in objects:
        if obj['score'] < class_thresholds[obj['label']]:
            continue

        cropped_table = {}

        bbox = obj['bbox']
        bbox = [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding]

        cropped_img = img.crop(bbox)

        table_tokens = [token for token in tokens if box_iou(token['bbox'], bbox) >= 0.5]
        for token in table_tokens:
            token['bbox'] = [token['bbox'][0] - bbox[0],
                             token['bbox'][1] - bbox[1],
                             token['bbox'][2] - bbox[0],
                             token['bbox'][3] - bbox[1]]

        # If table is predicted to be rotated, rotate cropped image and tokens/words:
        if obj['label'] == 'table rotated':
            cropped_img = cropped_img.rotate(270, expand=True)
            cropped_table["rotate"] = True
            for token in table_tokens:
                bbox = token['bbox']
                bbox = [cropped_img.size[0] - bbox[3] - 1,
                        bbox[0],
                        cropped_img.size[0] - bbox[1] - 1,
                        bbox[2]]
                token['bbox'] = bbox
        else:
            cropped_table["rotate"] = False

        cropped_table['image'] = cropped_img
        cropped_table['tokens'] = table_tokens


        table_crops.append(cropped_table)

    return table_crops


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score),
                            'bbox': [float(elem) for elem in bbox]})

    return objects


def get_cell_coordinates_by_row(table_data):
    # Extract rows and columns
    rows = [entry for entry in table_data if entry['label'] == 'table row']
    columns = [entry for entry in table_data if entry['label'] == 'table column']

    # Sort rows and columns by their Y and X coordinates, respectively
    rows.sort(key=lambda x: x['bbox'][1])
    columns.sort(key=lambda x: x['bbox'][0])

    # Function to find cell coordinates
    def find_cell_coordinates(row, column):
        cell_bbox = [column['bbox'][0], row['bbox'][1], column['bbox'][2], row['bbox'][3]]
        return cell_bbox

    # Generate cell coordinates and count cells in each row
    cell_coordinates = []

    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = find_cell_coordinates(row, column)
            row_cells.append({'column': column['bbox'], 'cell': cell_bbox})

        # Sort cells in the row by X coordinate
        row_cells.sort(key=lambda x: x['column'][0])

        # Append row information to cell_coordinates
        cell_coordinates.append({'row': row['bbox'], 'cells': row_cells, 'cell_count': len(row_cells)})

    # Sort rows from top to bottom
    cell_coordinates.sort(key=lambda x: x['row'][1])

    return cell_coordinates


def apply_ocr(cell_coordinates, cropped_table, reader, rotate=False):
    # let's OCR row by row
    data = dict()
    max_num_columns = 0
    for idx, row in enumerate(cell_coordinates):
        row_text = []

        maxx, maxy = 0, 0
        cells = []
        for cell in row["cells"]:
            cell = np.array(cropped_table.crop(cell["cell"]).rotate(-270 if rotate else 0, Image.NEAREST, expand = 1))
            maxx = max(maxx, cell.shape[1])
            maxy = max(maxy, cell.shape[0])
            cells.append(cell)

        for i in range(len(cells)):
            cells[i] = cv2.copyMakeBorder(cells[i], 0, maxy - cells[i].shape[0], 0, maxx - cells[i].shape[1],
                                          cv2.BORDER_CONSTANT, value=[255, 255, 255])

        if len(cells) == 0:
            return data

        result = reader.readtext_batched(cells)

        for r in result:
            if len(r) > 0:
                # print([x[1] for x in list(result)])
                text = " ".join([x[1] for x in r])
                row_text.append(text)

        if len(row_text) > max_num_columns:
            max_num_columns = len(row_text)

        data[idx] = row_text

    # pad rows which don't have max_num_columns elements
    # to make sure all rows have the same number of columns
    for row, row_data in data.copy().items():
        if len(row_data) != max_num_columns:
            row_data = row_data + ["" for _ in range(max_num_columns - len(row_data))]
        data[row] = row_data

    return data


def apply_ocr_old(cell_coordinates, cropped_table, reader):
    # let's OCR row by row
    data = dict()
    max_num_columns = 0
    for idx, row in enumerate(cell_coordinates):
        row_text = []
        for cell in row["cells"]:
            # crop cell out of image
            cell_image = np.array(cropped_table.crop(cell["cell"]))
            # apply OCR
            result = reader.readtext(np.array(cell_image), workers=4, decoder="beamsearch")
            if len(result) > 0:
                # print([x[1] for x in list(result)])
                text = " ".join([x[1] for x in result])
                row_text.append(text)

        if len(row_text) > max_num_columns:
            max_num_columns = len(row_text)

        data[idx] = row_text

    # print("Max number of columns:", max_num_columns)

    # pad rows which don't have max_num_columns elements
    # to make sure all rows have the same number of columns
    for row, row_data in data.copy().items():
        if len(row_data) != max_num_columns:
            row_data = row_data + ["" for _ in range(max_num_columns - len(row_data))]
        data[row] = row_data

    return data


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    detection_transform = transforms.Compose([
        MaxResize(800),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm")
    model.to(device)

    # update id2label to include "no object"
    id2label = model.config.id2label
    id2label[len(model.config.id2label)] = "no object"

    # new v1.1 checkpoints require no timm anymore
    structure_model = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-structure-recognition-v1.1-all")
    structure_model.to(device)

    structure_transform = transforms.Compose([
        MaxResize(1000),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    batch_size = 64
    reader = easyocr.Reader(['en'])

    structure_id2label = structure_model.config.id2label
    structure_id2label[len(structure_id2label)] = "no object"


    path = Path(".\data\processed_data")
    for folder in path.glob("*"):
        if not (folder / "converted_output_test_ocr.json").exists():
            with open(folder / "converted_output_test.json", "r") as file:
                data = json.load(file)

            with open(folder / "converted_output_test_ocr.json", 'w') as f:
                json.dump(data, f, indent=4)
        else:
            with open(folder / "converted_output_test_ocr.json", "r") as file:
                data = json.load(file)

        for j in trange(len(data), desc=f"Processing data - {folder.stem}"):
            if "ocr" in data[j]:
                continue
            image = data[j]["image"]
            # image = "/mnt/NVMe1T/MMFM-Challenge/data/phase2_data/mydoc/images/0c857eefdda4cf25f2a2692e05d4b4c2a70a9e4b961ff2a027db9ca3b3eae7d1.png"
            if image.endswith(".gif"):
                tmpi = np.asarray(Image.open(image).convert("RGB")).copy()
            else:
                tmpi = cv2.imread(image).copy()
            try:
                image = Image.fromarray(tmpi).convert("RGB")
            except Exception as e:
                print(image)
                raise e

            pixel_values = detection_transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(pixel_values)

            objects = outputs_to_objects(outputs, image.size, id2label)

            tokens = []
            detection_class_thresholds = {
                "table": 0.5,
                "table rotated": 0.5,
                "no object": 10
            }
            crop_padding = 10

            tables_crops = objects_to_crops(image, tokens, objects, detection_class_thresholds, padding=0)

            tables = []
            # tmp_res = table_engine(tmpi)

            for k, table in enumerate(tables_crops):
                cropped_table = table['image'].convert("RGB")
                pixel_values = structure_transform(cropped_table).unsqueeze(0).to(device)
                outputs = structure_model(pixel_values)
                cells = outputs_to_objects(outputs, cropped_table.size, structure_id2label)
                cell_coordinates = get_cell_coordinates_by_row(cells)
                data2 = apply_ocr(cell_coordinates, cropped_table, reader, table["rotate"])
                df = pd.DataFrame.from_dict(data2, orient="index")
                # df = img_byte_arr[0].df
                if df.empty:
                    continue
                new_header = df.iloc[0]  # grab the first row for the header
                df = df[1:]  # take the data less the header row
                df.columns = new_header  # set the header row as the df header
                html = re.sub(r'<tr.*>', '<tr>', df.to_html().replace('border="1" ', '')).replace(' class="dataframe"',
                                                                                                  "")
                html = minify_html.minify(html)

                tables.append(
                    [
                        [int(x) for x in objects[k]["bbox"]],
                        html,
                        1
                    ]
                )
                tmpi = cv2.rectangle(tmpi.copy(), (tables[-1][0][0], tables[-1][0][1]), (tables[-1][0][2], tables[-1][0][3]),
                                     (0, 255, 0),
                                     -1)

            # Image.fromarray(tmpi).show()
            # Image.fromarray(tmpi).save(fp.name, format="PNG")
            result = reader.readtext(tmpi, batch_size=batch_size, workers=4, decoder="beamsearch")

            new_results = []
            for i in range(len(result)):
                result[i] = list(result[i])
                result[i][-1] = float(result[i][-1])
                maxx = max(int(result[i][0][j][0]) for j in range(4))
                minx = min(int(result[i][0][j][0]) for j in range(4))
                miny = min(int(result[i][0][j][1]) for j in range(4))
                maxy = max(int(result[i][0][j][1]) for j in range(4))
                if len(tables) > 0:
                    used = 0
                    for k, table in enumerate(tables):
                        if minx >= table[0][0] and miny >= table[0][1]:
                            new_results.append(table)
                            used += 1
                        else:
                            break
                    for k in range(used):
                        tables.pop(0)
                result[i][0] = [minx, miny, maxx, maxy]
                new_results.append(result[i])

            for table in tables:
                new_results.append(table)

            data[j]["ocr"] = new_results

            with open(folder / "converted_output_test_ocr.json", 'w') as f:
                json.dump(data, f, indent=4)


if __name__ == "__main__":
    with keep.running():
        main()