from mefm.model import ModifiedEFM
import numpy as np
import pandas as pd
from cornac.eval_methods.base_method import BaseMethod


def get_predictions_df(model: ModifiedEFM, eval_data: BaseMethod) -> pd.DataFrame:
    """
    Get datasets with score predictions by specified Modified EFM.

    :param model: trained MEFM instance
    :param eval_data: data object that contains train/test split and additional modalities
    :return: pandas DataFrame with predicted scores
    """
    test_uids = eval_data.test_set.user_indices
    user_id = np.empty(0, dtype=int)
    item_id = np.empty(0, dtype=int)
    prediction = np.empty(0, dtype=float)
    for uid in test_uids:
        # predicting scores
        _, scores = model.rank(uid)

        # get known positives from train interactions
        known_positives = eval_data.train_set.csr_matrix.getrow(uid).indices

        # making known positives a really low number
        scores[known_positives] = 0

        # sort predictions
        sorted_indices = np.argsort(-scores)[:20]
        sorted_scores = scores[sorted_indices]

        user_id = np.append(user_id, np.repeat(uid, 20))
        item_id = np.append(item_id, sorted_indices)
        prediction = np.append(prediction, sorted_scores)

    return pd.DataFrame(dict(user_id=user_id, item_id=item_id, prediction=prediction))
