import json
import os
from typing import List
import pandas as pd

from models import ContentUnit
from text_utils import transform_html_to_source


def content_unit_to_dict(content_unit, parent_id=None):
    exclude_keys = {'replies'}  # Add keys you want to exclude here
    data = vars(content_unit)

    data = {k: v for k, v in data.items() if k not in exclude_keys}

    rows = [data]
    for reply in content_unit.replies:
        rows.extend(content_unit_to_dict(reply, parent_id=content_unit.id))
    return rows



def load_json_to_dataframe(json_file_path: str) -> pd.DataFrame:
    with open(json_file_path, 'r') as f:
        content_data = json.load(f)

    if 'replies' in content_data and content_data['replies'] is not None:
        content_data['replies'] = [reply for reply in content_data['replies'] if reply is not None]

    content_unit = ContentUnit(**content_data)
    rows = content_unit_to_dict(content_unit)
    df = pd.DataFrame(rows)
    return df

def find_all_json_files(directory: str) -> List[str]:
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files


def get_all_files():
    saved_files = find_all_json_files('/Users/tristandelforge/Documents/arguments/raw_arguments')
    dfs = [load_json_to_dataframe(i) for i in saved_files]
    df = pd.concat(dfs)
    return df


def get_train_df():
    df = get_all_files()
    return df.copy() #TODO: add proper split


def get_source_text_list(df: pd.DataFrame, min_len = 50):
    df['source_calculated'] = df['text'].apply(transform_html_to_source)
    df = df[df['source_calculated'].astype(str).str.len() >= min_len]
    return df['source_calculated'].tolist()

