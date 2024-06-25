import json
import os
import random
from typing import List
import pandas as pd

from common import generate_chain_aware_df_file_loc
from models import ContentUnit
from text_utils import transform_html_to_source, transform_html_to_paragraphs


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


def get_all_files(max_docs = None):
    if max_docs == None:
        saved_files = find_all_json_files('/Users/tristandelforge/Documents/arguments/raw_arguments')
    else:
        saved_files = find_all_json_files('/Users/tristandelforge/Documents/arguments/raw_arguments')
        random.shuffle(saved_files)
        saved_files = saved_files[:max_docs]

    dfs = [load_json_to_dataframe(i) for i in saved_files]
    df = pd.concat(dfs)
    return df


def get_train_df(max_docs = None):
    df = get_all_files(max_docs = max_docs)
    print(df.columns.tolist())
    print(f'read files: {df.shape}')
    return df.copy() #TODO: add proper split



def get_paragraph_text_list(df: pd.DataFrame, min_len = 50, max_len=1000):
    text_list = [paragraph for sublist in df['text'].apply(transform_html_to_paragraphs) for paragraph in sublist]
    title_list = df[~df['title'].isna()]['title'].tolist()
    l = sorted(list(set(text_list + title_list)))
    l = [i for i in l if len(i) > min_len and len(i) < max_len]
    return l


def get_html_text_list(df: pd.DataFrame, min_len = 50, max_len=1000):
    df['source_calculated'] = df['text']
    df = df[df['source_calculated'].astype(str).str.len() >= min_len]
    df = df[df['source_calculated'].astype(str).str.len() <= max_len]
    return df['source_calculated'].tolist()


def get_source_text_list(df: pd.DataFrame, min_len = 50, max_len=1000):
    df2 = df[~df['title'].isna()].copy()
    df2['source_calculated'] = df2['title'].astype(str) + ' \n ' + df2['text'].astype(str).apply(transform_html_to_source)
    df2=df2[df2['source_calculated'].astype(str).str.len() >= min_len]
    df2=df2[df2['source_calculated'].astype(str).str.len() <= max_len]

    df = df[df['title'].isna()].copy()
    df['source_calculated'] = df['text'].apply(transform_html_to_source)
    df = df[df['source_calculated'].astype(str).str.len() >= min_len]
    df = df[df['source_calculated'].astype(str).str.len() <= max_len]
    return sorted(list(set(df2['source_calculated'].tolist() +  df['source_calculated'].tolist())))


def get_comment_chain_text_list(df: pd.DataFrame, min_len = 50, max_len=1000):
    chains = build_comment_chains(df)

    chains2 = list()

    for chain in chains:
        t = f'{chain[0]["user"]} says:'

        for c in chain:
            f'{c["user"]} says:'
            if not pd.isnull(c['title'] ):
                t += f'\n {c["title"]}'
            t += f'\n {transform_html_to_source(c["text"])}'
        chains2.append(t)

    chains2 = [i for i in chains2 if len(i) <= max_len and len(i) >= min_len ]
    return chains2


def build_comment_chains(df):
    children_dict = df.groupby('parent_id')['id'].apply(list).to_dict()

    def get_children_ids(post_id):
        return children_dict.get(post_id, [])

    df['children_ids'] = df['id'].apply(get_children_ids)
    df['child_count'] = df['children_ids'].apply(len)


    leaf_node_ids = df[df['children_ids'].apply(len) == 0]['id']

    def get_chain(node_id):
        # Construct the chain by traversing back to the root
        chain = []
        current_id = node_id
        while True:
            node = df[df['id'] == current_id].iloc[0]
            chain.append(node.to_dict())
            if 'is_post' in node and node['is_post']:
                break
            current_id = node['parent_id']
        return chain[::-1]  # Reverse to start from the root to the leaf

    # Collect all chains starting from each leaf node
    all_chains = [get_chain(leaf_id) for leaf_id in leaf_node_ids]

    return all_chains


def generate_chain_aware_df():
    chains = build_comment_chains(get_train_df())

    chain_aware_dfs = list()
    for c in chains:
        chain_id = c[-1]['id']
        chain_df = pd.DataFrame.from_dict(c)
        chain_df['chain_id'] = chain_id
        chain_df['chain_rank'] = chain_df.index.copy()
        chain_aware_dfs.append(chain_df)

    chain_aware_df = pd.concat(chain_aware_dfs)
    chain_aware_df.to_csv(generate_chain_aware_df_file_loc, index = False)


def get_a_common_chain_df_to_llm_input_by_chain_id(df,
                                                   chain_id,
                                                   content_id):
    c_df = df[df['chain_id'] == chain_id]
    chain_rank = c_df[c_df['id'] == content_id].iloc[0]['chain_rank']
    c_df = c_df[c_df['chain_rank'] <= chain_rank]

    output_text = f'Post by {c_df.iloc[0]["user"]}, title: {c_df.iloc[0]["title"]}, initial comment: {transform_html_to_source(c_df.iloc[0]["text"])} \n'

    for idx, row in c_df.iloc[1:].iterrows():
        output_text+= f'Responded to by {row["user"]}, comment: {transform_html_to_source(row["text"])} \n'

    return output_text


if __name__ == '__main__':
    df = fill_in_common_chain_df_with_llm_input()
    df
    # generate_chain_aware_df()
    # generate_chain_aware_df()
    # # df = get_train_df(max_docs = 50)
    # # df2 = analyze_comment_chains(df)
    # # chains = build_comment_chains(df)
    # # chains
    #
    # a = get_comment_chain_text_list(get_train_df(max_docs = 50))
    # a
    #
