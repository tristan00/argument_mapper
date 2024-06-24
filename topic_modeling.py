import json
import os
from typing import List, Tuple

from gensim.corpora import Dictionary
from gensim.models import LdaModel, LdaMulticore
from gensim.test.utils import datapath, get_tmpfile

import common
from dataset_utils import get_source_text_list, get_train_df
from text_utils import tokenize_doc

chunksize = 1000
passes = 10
iterations = 1000
eval_every = None



def train_lda(
    docs: List[List[str]],
    model_dir_location: str,
lda_num_topics = 50
):
    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=5, no_above=0.2)

    corpus = list()
    for doc in docs:
        corpus.append(dictionary.doc2bow(doc[:common.lda_max_text_length]))

    print("Number of unique tokens: %d" % len(dictionary))
    print("Number of documents: %d" % len(corpus))

    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token


    model = LdaMulticore(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        eta="auto",
        iterations=iterations,
        num_topics=lda_num_topics,
        passes=passes,
        eval_every=eval_every,
        workers=os.cpu_count()
    )

    top_topics = model.top_topics(corpus)
    avg_topic_coherence = sum([t[1] for t in top_topics]) / lda_num_topics
    print("Average topic coherence: %.4f." % avg_topic_coherence)

    temp_file = datapath(common.lda_model_loc)
    model.save(temp_file)

    temp_file = datapath(common.dictionary_loc)
    dictionary.save(temp_file)

    return avg_topic_coherence


def train_lda_on_source_messages(
lda_num_topics = 50
):
    train_docs = get_source_text_list(get_train_df())
    train_docs = [tokenize_doc(doc) for doc in train_docs]
    return train_lda(
        docs=train_docs,
        model_dir_location=f"{common.main_dir}/lda_model",
        lda_num_topics=lda_num_topics
    )


def load_lda_model(
) -> Tuple[LdaModel, Dictionary]:
    tmp_fname_dict = get_tmpfile(f"{common.main_dir}/dictionary")
    tmp_fname_model = get_tmpfile(f"{common.main_dir}/lda_model")

    loaded_dct = Dictionary.load(tmp_fname_dict)
    loaded_dct: Dictionary
    loaded_model = LdaModel.load(tmp_fname_model)
    return loaded_model, loaded_dct




if __name__ == '__main__':
    results = dict()
    for i in range(3, 100):
        results[i] =  train_lda_on_source_messages(
            lda_num_topics=i
        )
        print(results)