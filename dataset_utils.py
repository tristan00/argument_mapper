import json
import os
import random
from typing import List
import pandas as pd

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
    df = analyze_comment_chains(df)
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


def analyze_comment_chains(df):
    children_dict = df.groupby('parent_id')['id'].apply(list).to_dict()

    def get_children_ids(post_id):
        return children_dict.get(post_id, [])

    # def check_deleted(post_id):
    #     current_id = post_id
    #     while True:
    #         current_text = df.loc[df['id'] == current_id, 'text'].values[0]
    #         if "[deleted]" in current_text:
    #             return True
    #         if df.loc[df['id'] == current_id, 'is_post'].values[0]:
    #             break  # Stop if we reach the root post
    #         current_id = df.loc[df['id'] == current_id, 'parent_id'].values[0]
    #     return False

    # Initialize new columns
    df['children_ids'] = df['id'].apply(get_children_ids)
    df['child_count'] = df['children_ids'].apply(len)
    # df['deleted_conversation'] = df['id'].apply(check_deleted)

    # Calculate depth and maximum depth of the conversation chain

    # def get_depth(post_id):
    #     if df.loc[df['id'] == post_id, 'is_post'].values[0]:
    #         return 0
    #     else:
    #         parent_id = df.loc[df['id'] == post_id, 'parent_id'].values[0]
    #         return 1 + get_depth(parent_id)
    #
    # def max_depth_of_the_conversation_chain(post_id):
    #     # Base case: if no children, the max depth from this node is just its own depth
    #     if post_id not in children_dict or not children_dict[post_id]:
    #         # Directly return the depth of this node since it has no children
    #         return df.loc[df['id'] == post_id, 'depth_in_conversation_chain'].values[0]
    #
    #     # Recursive case: find the maximum depth among all children
    #     max_depth = 0
    #     for child_id in children_dict[post_id]:
    #         # Calculate the depth of each child relative to this node
    #         child_depth = max_depth_of_the_conversation_chain(child_id)
    #         # Compare it to the current max depth, keep the maximum
    #         max_depth = max(max_depth, child_depth)
    #
    #     return max_depth
    #
    # df['depth_in_conversation_chain'] = df['id'].apply(lambda x: get_depth(x))
    # df['max_depth_of_the_conversation_chain'] = df['id'].apply(lambda x: max_depth_of_the_conversation_chain(x))
    return df


def build_comment_chains(df):
    # Identify leaf nodes as those without children
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




if __name__ == '__main__':
    # df = get_train_df(max_docs = 50)
    # df2 = analyze_comment_chains(df)
    # chains = build_comment_chains(df)
    # chains

    a = get_comment_chain_text_list(get_train_df(max_docs = 50))
    a
