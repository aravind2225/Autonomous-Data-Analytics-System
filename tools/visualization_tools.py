import pandas as pd


GLOBAL_DF = None


def set_dataframe(df):
    """
    Store dataframe globally for visualization workflow.
    """

    global GLOBAL_DF
    GLOBAL_DF = df.copy()


def get_dataframe():
    """
    Return dataframe.
    """

    global GLOBAL_DF
    return GLOBAL_DF


def get_numeric_columns():

    global GLOBAL_DF

    if GLOBAL_DF is None:
        return []

    return list(
        GLOBAL_DF.select_dtypes(include='number').columns
    )


def get_categorical_columns():

    global GLOBAL_DF

    if GLOBAL_DF is None:
        return []

    return list(
        GLOBAL_DF.select_dtypes(exclude='number').columns
    )