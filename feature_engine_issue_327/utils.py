import logging

import numpy as np
import pandas as pd


def correlated_feat_removal(X, cols=None, corr_thresh=0.8):
    """
    Remove correlated features from a pandas dataframe.

    Parameters
    ----------
    X: pandas dataframe of shape = [n_samples, n_features]
    cols: string. if None, then the function will make the selection among all features.
    corr_thresh: float, default=0.8
        The correlation threshold above which a feature will be deemed correlated with
        another one and removed from the dataset.
    """
    # Get the column list
    cols = cols if cols else X.columns

    # Calculate correlation matrix, take the absolute value
    corr_matrix = X[cols].corr().abs()

    # The matrix is symmetric so we need to extract upper triangle matrix without diagonal (k = 1)
    corr_upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find index of feature columns with correlation greater than threshold
    feats_to_drop = [col for col in corr_upper.columns if any(corr_upper[col] > corr_thresh)]

    # Get the selected feature list
    selected_cols = [col for col in corr_upper.columns if col not in feats_to_drop]

    # Logg the size of the selected features
    logging.info(f"Selected # of columns: {len(selected_cols)}")

    return corr_upper, selected_cols


def get_corr_matix(X, cols=None):
    """
    Calculate the correlation matrix using pandas .corr().

    Parameters
    ----------
    X: pandas dataframe of shape = [n_samples, n_features]
    cols: string. if None, then the function will make the selection among all features.
    """
    # Get the column list
    cols = cols if cols else X.columns

    # Calculate the correlation matrix
    corr_matrix = X[cols].corr("pearson").abs()

    # The matrix is symmetric so we extract upper triangle matrix without diagonal (k = 1)
    corr_upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    return corr_upper


def get_second_matrix(X, cols=None, matrix_type="std"):
    """
    Calculate the second matrix which will be used as the selection criteria.

    Parameters
    ----------
    X: pandas dataframe of shape = [n_samples, n_features]
    cols: list. if None, then the function will make the selection among all features.
    matrix_type: string. matrix_type can choose from one of these ["std", "missings"]
    """
    # Get the column list
    cols = cols if cols else X.columns

    if matrix_type == "std":
        matrix = X[cols].std()
    elif matrix_type == "missings":
        matrix = X[cols].isnull().sum()
    else:
        logging.info("please select from one of these ['std, 'missings']")

    diff_matrix = pd.DataFrame(list(matrix) - np.array(list(matrix)).reshape(-1, 1), columns=cols, index=cols)
    diff_matrix = diff_matrix.where(np.triu(np.ones(diff_matrix.shape), k=1).astype(bool))
    return diff_matrix


def get_corr_mask(corr, corr_thresh=0.8):
    """
    Mask the numbers below threshold in the corrlation matrix as 0.

    Parameters
    ----------
    corr: pandas dataframe of shape = [n_features, n_features]
    corr_thresh: float, default=0.8
        The correlation threshold above which a feature will be deemed correlated with
        another one and removed from the dataset.
    """
    corr_mask = corr.copy()
    corr_mask[corr_mask < corr_thresh] = 0
    return corr_mask


def get_matrix_mask(matrix, threshold=0, mask_value=1):
    """
    Mask the numbers in a given matrix.

    Parameters
    ----------
    matrix: pandas dataframe of shape = [n_features, n_features]
    threshold: float, default=0
        The correlation threshold above which a feature will be deemed correlated with
        another one and removed from the dataset.
    mask_value float, default=1
    """
    matrix_mask = matrix.copy()
    # When looking along the columns.
    # When a value is less than 0, it means the variance (std) of feature A in the column is
    # smaller than feature B in the indx
    # Then we want to drop feature A, we assign number 1 to mark it
    matrix_mask[matrix_mask <= threshold] = -1 * mask_value
    matrix_mask[matrix_mask > threshold] = 1 * mask_value
    return matrix_mask


def make_selection(corr_mask, matrix_mask, cols=None, corr_thresh=0.8):
    """Combine two selection criterias by  multipling them togther, and then make the selection based on results.

    Parameters
    ----------
    corr_mask: pandas dataframe of shape = [n_features, n_features],
               corr_mask is filled in with either 0 or correaltion values higher than threhold.
    matrix_mask: pandas dataframe of shape = [n_features, n_features]
               matrix_mask is filled in with either 1 or -1 to indicate if one feature in columns has higher
               std/missing values than another in the rows/index.

    """
    # Get the column list
    cols = cols if cols else corr_mask.columns

    # First combine two matrix by multipling their mask matrix together
    mask_union = matrix_mask * corr_mask

    # Make the first round of selection along the column direction
    feats_to_drop_along_col = [col for col in cols if any(mask_union[col] < -1 * corr_thresh)]
    logging.info(f"{len(feats_to_drop_along_col)} features dropped along the column: {feats_to_drop_along_col}")

    # Make the second round of selection along the row(index) direction
    feats_to_drop_along_row = [
        col for col in cols if any(mask_union.T[col] > corr_thresh) and col not in feats_to_drop_along_col
    ]
    logging.info(f"{len(feats_to_drop_along_row)} features dropped along the row: {feats_to_drop_along_row}")

    # Get the list of feature to be dropped and kept
    feats_to_drop = feats_to_drop_along_col + feats_to_drop_along_row
    feats_to_keep = [col for col in cols if col not in feats_to_drop]

    return feats_to_keep, feats_to_drop
