from typing import List, Dict

import pandas as pd
import numpy as np
from catboost import CatBoostRanker, Pool
from itertools import product, combinations


def generate_interactions(df, numeric_cols):
    df_c = df.copy()
    for col1, col2 in combinations(numeric_cols, 2):
        df_c[f'{col1}_x_{col2}'] = df_c[col1] * df_c[col2]
    return df_c

def general_case(score_df: pd.DataFrame, set_inputs: Dict, score_column: str, parameter_columns: List[str], iterations: int) -> pd.DataFrame:
    fit_df = score_df.copy()

    fit_df['group_id'] = 0

    numeric_cols = fit_df[parameter_columns].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = fit_df[parameter_columns].select_dtypes(include=['object', 'category']).columns.tolist()

    fit_df = generate_interactions(fit_df, numeric_cols)

    interaction_cols = [f'{col1}_x_{col2}' for col1, col2 in combinations(numeric_cols, 2)]
    updated_parameter_columns = numeric_cols + categorical_cols + interaction_cols

    train_pool = Pool(data=fit_df[updated_parameter_columns], label=fit_df[score_column], group_id=fit_df['group_id'], cat_features=categorical_cols)

    model = CatBoostRanker(iterations=100, learning_rate=0.1, depth=12, loss_function='YetiRank')
    model.fit(train_pool)

    synthetic_data = pd.DataFrame(index=range(iterations))
    for col in parameter_columns:
        if col in set_inputs:
            synthetic_data[col] = [set_inputs[col]] * iterations
        elif col in categorical_cols:
            synthetic_data[col] = np.random.choice(score_df[col].dropna().unique(), iterations)
        else:
            min_val, max_val = score_df[col].min(), score_df[col].max()
            synthetic_data[col] = np.random.uniform(min_val, max_val, iterations)

    extended_df = pd.concat([fit_df, synthetic_data], ignore_index=True)
    extended_df = generate_interactions(extended_df, numeric_cols)
    extended_df['group_id'] = 0

    extended_pool = Pool(data=extended_df[updated_parameter_columns], cat_features=categorical_cols)
    extended_df['predicted_score'] = model.predict(extended_pool)
    sorted_df = extended_df.sort_values(by='predicted_score')

    sorted_df = sorted_df[parameter_columns + [score_column, 'predicted_score']]
    return sorted_df
