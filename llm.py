import transformers
import torch
from transformers import logging
import pandas as pd
from common import llm_model_id, generate_chain_aware_df_file_loc
from dataset_utils import get_a_common_chain_df_to_llm_input_by_chain_id
from llm_prompts import p1

logging.set_verbosity_debug()


def get_pipeline():
    pipeline = transformers.pipeline(
        "text-generation", model=llm_model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto", max_length=10000
    )
    pipeline("Hey how are you doing today?")
    return pipeline


def fill_in_common_chain_df_with_llm_output():
    df = pd.read_csv(generate_chain_aware_df_file_loc, nrows = 100)
    llm_pipeline = get_pipeline()

    for idx, row in df.iterrows():
        gpt_input = get_a_common_chain_df_to_llm_input_by_chain_id(df, row['chain_id'], row['id'])
        input_text = f'f{p1} \n {gpt_input}'
        import time
        start_time = time.time()
        output = llm_pipeline(input_text)
        print(output)
        print(f'Done: {time.time() - start_time}, {len(input_text)}')


if __name__ == '__main__':
    fill_in_common_chain_df_with_llm_output()

