import json
import os
import random
import time
from typing import List, Tuple
import pandas as pd
from gensim.models.phrases import Phrases, Phraser
from gensim.models.word2vec import LineSentence

from gensim.corpora import Dictionary
from gensim.matutils import hellinger
from gensim.models import LdaModel, LdaMulticore
from gensim.test.utils import datapath, get_tmpfile
import numpy as np
import common
from dataset_utils import get_source_text_list, get_train_df, get_html_text_list, get_paragraph_text_list, \
    get_comment_chain_text_list
from text_utils import tokenize_doc

eval_every = None


def train_lda(

        docs: List[List[str]],
        model_dir_location: str,
        lda_num_topics=50,
        chunksize=1000,
        passes=10,
        iterations=1000,
        decay=.8,
        alpha='asymmetric',
        eta='auto',
        no_below=5,
        no_above=0.2,
        phraser_min_count=1,
        phraser_threshold=10,
        phraser_scorer='default',
        phraser_threshold_npmi=0.0
):
    start_time = time.time()

    if phraser_scorer == 'default':
        phrases = Phrases(docs, min_count=phraser_min_count,
                          threshold=phraser_threshold,
                          scoring=phraser_scorer)
    else:
        phrases = Phrases(docs, min_count=phraser_min_count,
                          threshold=phraser_threshold_npmi,
                          scoring=phraser_scorer)
    ngram_phraser = Phraser(phrases)
    docs_with_ngrams = [ngram_phraser[doc] for doc in docs]

    dictionary = Dictionary(docs_with_ngrams)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)

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
        eta=eta,
        iterations=iterations,
        num_topics=lda_num_topics,
        passes=passes,
        eval_every=eval_every,
        workers=os.cpu_count(),
        decay=decay,
        alpha=alpha
    )
    end_time = time.time()

    model2 = LdaMulticore(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        eta=eta,
        iterations=iterations,
        num_topics=lda_num_topics,
        passes=passes,
        eval_every=eval_every,
        workers=os.cpu_count(),
        decay=decay,
        alpha=alpha
    )

    top_topics = model.top_topics(corpus)
    avg_topic_coherence = sum([t[1] for t in top_topics]) / lda_num_topics
    perplexity = model.log_perplexity(corpus)

    def topic_diversity(topics):
        unique_tokens = set()
        total_tokens = sum(len(topic) for topic in topics)
        for topic in topics:
            unique_tokens.update(topic)
        return len(unique_tokens) / total_tokens

    topics = [[word for word, _ in model.show_topic(topicid, topn=10)] for topicid in range(lda_num_topics)]
    diversity = topic_diversity(topics)

    def entropy_of_distribution(distribution):
        probabilities = [prob for _, prob in distribution]
        return -np.sum([p * np.log(p) if p > 0 else 0 for p in probabilities])

    entropy_scores = [entropy_of_distribution(doc) for doc in model[corpus]]
    avg_entropy = sum(entropy_scores) / len(entropy_scores)

    def topic_stability(model1, model2, num_topics):
        distances = []
        for t1 in range(num_topics):
            dists = [hellinger(model1.get_topic_terms(t1), model2.get_topic_terms(t2)) for t2 in range(num_topics)]
            distances.append(min(dists))
        return sum(distances) / num_topics

    stability_score = topic_stability(model, model2, lda_num_topics)

    print("Average topic coherence: %.4f." % avg_topic_coherence)
    print(f'perplexity: {perplexity}')
    print(f'diversity: {diversity}')
    print(f'avg_entropy: {avg_entropy}')
    print(f'stability_score: {stability_score}')

    temp_file = datapath(common.lda_model_loc)
    model.save(temp_file)

    temp_file = datapath(common.dictionary_loc)
    dictionary.save(temp_file)

    return {
        'avg_topic_coherence': avg_topic_coherence,
        'perplexity': perplexity,
        'diversity': diversity,
        'avg_entropy': avg_entropy,
        'stability_score': stability_score,
        'model_training_time': end_time - start_time
    }


def train_lda_on_source_messages(
        lda_num_topics,
        chunksize,
        passes,
        iterations,
        decay,
        alpha,
        eta,
        no_below=5,
        no_above=0.2,
        min_len=50,
        max_len=1000,
        phraser_min_count=1,
        phraser_threshold=10,
        phraser_scorer='default',
        phraser_threshold_npmi=0.0,
        dataset='post'
):
    print(dataset)
    if dataset == 'post':
        train_docs = get_source_text_list(get_train_df(), min_len=min_len, max_len=max_len)
    elif dataset == 'post_html':
        train_docs = get_html_text_list(get_train_df(), min_len=min_len, max_len=max_len)
    elif dataset == 'paragraph':
        train_docs = get_paragraph_text_list(get_train_df(), min_len=min_len, max_len=max_len)
    elif dataset == 'chain':
        train_docs = get_comment_chain_text_list(get_train_df(), min_len=min_len, max_len=max_len)


    else:
        raise Exception
    train_docs = [tokenize_doc(doc) for doc in train_docs]
    measurements = train_lda(
        docs=train_docs,
        model_dir_location=f"{common.main_dir}/lda_model",
        lda_num_topics=lda_num_topics,
        chunksize=chunksize,
        passes=passes,
        iterations=iterations,
        decay=decay,
        alpha=alpha,
        eta=eta,
        no_below=no_below,
        no_above=no_above,
        phraser_min_count=phraser_min_count,
        phraser_threshold=phraser_threshold,
        phraser_scorer=phraser_scorer,
        phraser_threshold_npmi=phraser_threshold_npmi

    )
    inputs = dict(
        lda_num_topics=lda_num_topics,
        chunksize=chunksize,
        passes=passes,
        iterations=iterations,
        decay=decay,
        alpha=alpha,
        eta=eta,
        no_below=no_below,
        no_above=no_above,
        min_len=min_len,
        max_len=max_len,
        phraser_min_count=phraser_min_count,
        phraser_threshold=phraser_threshold,
        phraser_scorer=phraser_scorer,
        phraser_threshold_npmi=phraser_threshold_npmi,
        dataset=dataset
    )

    inputs.update(measurements)
    return inputs


def load_lda_model(
) -> Tuple[LdaModel, Dictionary]:
    tmp_fname_dict = get_tmpfile(f"{common.main_dir}/dictionary")
    tmp_fname_model = get_tmpfile(f"{common.main_dir}/lda_model")

    loaded_dct = Dictionary.load(tmp_fname_dict)
    loaded_dct: Dictionary
    loaded_model = LdaModel.load(tmp_fname_model)
    return loaded_model, loaded_dct


def rank_results(path):
    df = pd.read_csv(path)

    # Ascending = False if a lower value should score higher (better)
    # avg_topic_coherence: High values indicate that the words within a topic frequently co-occur in your corpus, suggesting that the topic is meaningful and interpretable.
    # Perplexity: Low values of perplexity indicate a model that predicts the sample well, suggesting better general performance. This means the model is more sure of its topic assignments.
    # Diversity: High values indicate that the model has a wide range of unique words across different topics, which can suggest that the topics are varied and well-separated.
    # Entropy: Low values signify that documents have more definitive topic distributions, with some topics being very dominant, which can indicate clearer topic structure or more focused documents.
    # Stability: High values reflect that the topics are stable across different runs of the model with varying initializations or data subsets, suggesting robustness and reliability of the topic definitions.

    df['coherence_rank'] = df['avg_topic_coherence'].rank(ascending=False)
    df['perplexity_rank'] = df['perplexity'].rank(ascending=True)
    df['diversity_rank'] = df['diversity'].rank(ascending=False)
    df['entropy_rank'] = df['avg_entropy'].rank(ascending=True)
    df['stability_rank'] = df['stability_score'].rank(ascending=False)
    df['model_training_time_rank'] = df['model_training_time'].rank(ascending=False)

    weights = {
        'coherence_rank': 5,
        'perplexity_rank': 1,
        'diversity_rank': 3,
        'entropy_rank': 2,
        'stability_rank': 2,
        'model_training_time_rank': 1
    }

    df['weighted_score'] = (
            df['coherence_rank'] * weights['coherence_rank'] +
            df['perplexity_rank'] * weights['perplexity_rank'] +
            df['diversity_rank'] * weights['diversity_rank'] +
            df['entropy_rank'] * weights['entropy_rank'] +
            df['stability_rank'] * weights['stability_rank']
    )

    df = df.sort_values('weighted_score')
    return df


if __name__ == '__main__':
    rank_results(common.topic_modeling_param_results_save_loc)
    results = list()
    try:
        results = list(pd.read_csv(common.topic_modeling_param_results_save_loc).to_dict(orient='records'))
    except:
        import traceback

        traceback.print_exc()
    # results=list()

    for i in range(10000):
        try:
            lda_num_topics = random.randint(2, 25)
            chunksize = random.randint(100, 2000)
            passes = random.randint(1, 12)
            iterations = random.randint(1, 250)
            decay = random.uniform(0.5, 0.99)
            alpha_value = random.choice(['symmetric', 'asymmetric'])
            eta_value = random.choice(['symmetric', 'auto'])
            no_below = random.randint(4, 100)
            no_above = random.uniform(0.01, 0.99)
            min_len = random.randint(1, 100)
            max_len = random.randint(110, 3000)

            phraser_min_count = random.randint(1, 50)
            phraser_threshold = random.randint(2, 40)
            phraser_threshold_npmi = random.uniform(-.999, .999)
            phraser_scorer = random.choice(['npmi', 'default'])

            dataset = random.choice(['post_html', 'post', 'paragraph', 'chain'])

            result = train_lda_on_source_messages(
                lda_num_topics=lda_num_topics,
                chunksize=chunksize,
                passes=passes,
                iterations=iterations,
                decay=decay,
                alpha=alpha_value,
                eta=eta_value,
                no_below=no_below,
                no_above=no_above,
                min_len=min_len,
                max_len=max_len,
                phraser_min_count=phraser_min_count,
                phraser_threshold=phraser_threshold,
                phraser_scorer=phraser_scorer,
                phraser_threshold_npmi=phraser_threshold_npmi,
                dataset=dataset
            )
            results.append(result)
            print(result)
            print( pd.DataFrame.from_dict(results).shape)
            pd.DataFrame.from_dict(results).to_csv(common.topic_modeling_param_results_save_loc, index=False)

        except:
            import traceback
            traceback.print_exc()
            time.sleep(1)
