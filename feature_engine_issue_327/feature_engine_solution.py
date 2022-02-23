# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
from utils import (
    correlated_feat_removal,
    get_corr_mask,
    get_corr_matix,
    get_matrix_mask,
    get_second_matrix,
    make_selection,
)

# ### 1. Prepare a dataset

data = pd.DataFrame(
    [
        {"a": 1, "b": 2, "c": 3, "d": 2, "e": -1},
        {"a": 10, "b": 20, "c": 30, "d": 18, "e": -9},
        {"a": 20, "b": 40, "c": 10, "d": 38, "e": -18},
        {"a": 19.3, "b": 35, "c": 12, "d": 12, "e": -18},
    ]
)

data.head()

# ### 2. Correlated feature removal - commonly-used way
#
# The correlated features are removed following these steps:
#
# 1. We use pandas `.corr()` to get a corr_matrix and then check the correlation among features.
# 2. Then we check the column list one by one.
# 3. For each column, we calculate the correlations between this feature and all the rest. When any of the absolute
# value of the correlation excceds the threshold, we drop that feature.

corr, selected_cols = correlated_feat_removal(X=data, cols=None, corr_thresh=0.8)

corr

selected_cols

corr, selected_cols_double_check = correlated_feat_removal(X=data, cols=list(selected_cols), corr_thresh=0.8)

assert selected_cols == selected_cols_double_check, "The first round of correlated feature removal is not complete"

# ### 3. Feature selection based on different criteria, e.g. std, missings

# Get the correlation matrix and std difference matrix
corr = get_corr_matix(X=data)
std = get_second_matrix(X=data, matrix_type="std")
missing = get_second_matrix(X=data, matrix_type="missings")

# Mask both matrices
corr_mask = get_corr_mask(corr=corr, corr_thresh=0.8)
std_mask = get_matrix_mask(matrix=std, threshold=0, mask_value=1)
missing_mask = get_matrix_mask(matrix=missing, threshold=0, mask_value=1)

corr

std

std_mask

missing

missing_mask

# #### 3.1 Smart feature selection based on `correlation` and `std`
# When two feature are correlated, drop the one with lower standard deviation.

feats_to_keep, feats_to_drop = make_selection(corr_mask=corr_mask, matrix_mask=std_mask, corr_thresh=0.8)

feats_to_keep, feats_to_drop

# Check if we still have correlated features dataframe:

assert data[data[feats_to_keep].corr().abs() > 0.8].sum().sum() == 0.0, "There are still correlated features"

# #### 3.2 Smart feature selection based on `correlation` and `missings`
# When two feature are correlated, drop the one with less missing values.

feats_to_keep, feats_to_drop = make_selection(corr_mask=corr_mask, matrix_mask=missing_mask, corr_thresh=0.8)

feats_to_keep, feats_to_drop

# Check if we still have correlated features in the dataframe

assert data[data[feats_to_keep].corr().abs() > 0.8].sum().sum() == 0.0, "There are still correlated features"
