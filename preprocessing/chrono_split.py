import pandas as pd
from typing import Tuple


def chrono_split(
        data: pd.DataFrame,
        user_col: str,
        item_col: str,
        timestamp_col: str,
        val_threshold: str,
        test_threshold: str,
        min_train_ratings: int,
        min_test_ratings: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the data into train, validation and test sets chronologically.

    :param data: data to be split
    :param user_col: the name of column in the dataset that contains user ids
    :param item_col: the name of column in the dataset that contains item ids
    :param timestamp_col: the name of column in the dataset that contains timestamps
    :param val_threshold: the date from which the validation set begins
    :param test_threshold: the date from which the test set begins
    :param min_train_ratings: the minimum number of interactions for each user and item in train set
    :param min_test_ratings: the minimum number of interactions for each user in validation and test sets
    :return: train, validation and test datasets
    """
    # split the data
    test_transactions = data[data.loc[:, timestamp_col] >= test_threshold]
    val_transactions = data[
        (data.loc[:, timestamp_col] >= val_threshold) & (data.loc[:, timestamp_col] < test_threshold)]
    train_transactions = data[data.loc[:, timestamp_col] < val_threshold]

    # filter the train to ensure minimum interactions count for user and items
    min_user_count = 1
    min_item_count = 1
    temp = train_transactions.loc[:, [user_col, item_col]]
    while (min_user_count < min_train_ratings) or (min_item_count < min_train_ratings):
        correct_users = temp.groupby(user_col).count().query(f"{item_col} > {min_train_ratings - 1}").reset_index().loc[
                        :, user_col]
        temp = temp[temp.loc[:, user_col].isin(correct_users)]

        correct_items = temp.groupby(item_col).count().query(f"{user_col} > {min_train_ratings - 1}").reset_index().loc[
                        :, item_col]
        temp = temp[temp.loc[:, item_col].isin(correct_items)]

        min_user_count = temp.groupby(user_col).count().loc[:, item_col].min()
        min_item_count = temp.groupby(item_col).count().loc[:, user_col].min()
    train_transactions = pd.merge(train_transactions, temp, how="inner", on=[user_col, item_col])

    # remove users and items that do not appear in train from validation and test
    train_users = set(train_transactions.loc[:, user_col].unique().tolist())
    train_items = set(train_transactions.loc[:, item_col].unique().tolist())
    predictable_val_transactions = val_transactions[
        val_transactions.loc[:, user_col].isin(train_users) & \
        val_transactions.loc[:, item_col].isin(train_items)
        ]
    predictable_test_transactions = test_transactions[
        test_transactions.loc[:, user_col].isin(train_users) & \
        test_transactions.loc[:, item_col].isin(train_items)
        ]

    # calculate user transaction counts for each subset
    train_counts = train_transactions.groupby(user_col).count().loc[:, [item_col]].reset_index()
    val_counts = predictable_val_transactions.groupby(user_col).count().loc[:, [item_col]].reset_index()
    test_counts = predictable_test_transactions.groupby(user_col).count().loc[:, [item_col]].reset_index()

    # compare counts
    counts = pd.merge(
        pd.merge(train_counts, val_counts, how="left", on=user_col, suffixes=["_train", "_val"]),
        test_counts,
        how="left",
        on=user_col
    )
    counts.loc[:, "train_val_sum"] = (counts.loc[:, item_col + "_train"] + counts.loc[:, item_col + "_val"]).where(
        counts.loc[:, item_col + "_val"].isna() == False,
        counts.loc[:, item_col + "_train"]
    )
    valid_val_users = set(counts[
                              (counts.loc[:, item_col + "_val"] >= min_test_ratings) & \
                              (counts.loc[:, item_col + "_train"] >= counts.loc[:, item_col + "_val"])
                              ].loc[:, user_col])
    valid_test_users = set(counts[
                               (counts.train_val_sum >= counts.loc[:, item_col]) & (
                                           counts.loc[:, item_col] >= min_test_ratings) & \
                               (counts.loc[:, item_col + "_val"] >= min_test_ratings) & \
                               (counts.loc[:, item_col + "_train"] >= counts.loc[:, item_col + "_val"])
                               ].loc[:, user_col])

    valid_val_transactions = predictable_val_transactions[
        predictable_val_transactions.loc[:, user_col].isin(valid_val_users)
    ]
    valid_test_transactions = predictable_test_transactions[
        predictable_test_transactions.loc[:, user_col].isin(valid_test_users)
    ]

    return train_transactions, valid_val_transactions, valid_test_transactions
