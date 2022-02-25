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

import numpy as np
import pandas as pd
from feature_engine.selection import DropCorrelatedFeatures
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
        {"a": 1, "b": 2, "c": 3, "d": 2, "e": -1, "f": 3},
        {"a": 10, "b": 20, "c": 30, "d": np.NaN, "e": -9, "f": 4},
        {"a": 20, "b": 40, "c": 10, "d": 38, "e": -18, "f": -20},
        {"a": 19.3, "b": 35, "c": 12, "d": 12, "e": -18, "f": 15},
    ]
)

data.head()

# ### 2. Correlated feature removal - commonly-used way

# #### 2.1 Manually coded - commonly-used approach
# The correlated features are removed following these steps:
#
# 1. We use pandas `.corr()` to get a corr_matrix, i.e., correlation among features.
# 2. Then we check the column list one by one.
# 3. For each column, we calculate the correlations between this feature and all the rest. When any of the absolute
# value of the correlation excceds the threshold, we drop that feature.

corr, selected_cols = correlated_feat_removal(X=data, cols=None, corr_thresh=0.8)

corr

selected_cols

corr, selected_cols_double_check = correlated_feat_removal(X=data, cols=list(selected_cols), corr_thresh=0.8)

assert selected_cols == selected_cols_double_check, "The first round of correlated feature removal is not complete"

# #### 2.2 Use feature engine

# +
tr = DropCorrelatedFeatures(variables=None, method="pearson", threshold=0.8)
Xt = tr.fit_transform(data)

tr.correlated_feature_sets_
Xt.columns
# -

# Same results for both, good!

# ### 3. Feature selection based on different criteria, e.g. std, missings

# Get the correlation matrix and std difference matrix
corr = get_corr_matix(X=data)
std_diff = get_second_matrix(X=data, second_matrix_type="std")
missing_diff = get_second_matrix(X=data, second_matrix_type="missings")

# Mask both matrices
corr_mask = get_corr_mask(corr=corr, corr_thresh=0.8)
std_mask = get_matrix_mask(matrix=std_diff, threshold=0, mask_value=1)
missing_mask = get_matrix_mask(matrix=missing_diff, threshold=0, mask_value=1)

# Correlation among features
corr

# Difference of std among features
std_diff

# Difference of num of missings  among features
missing_mask

corr_mask

# +
std_mask
# Take two examples:
# Eaxmple 1: For pair (index_a, col_b), the std_mask is 1, meaning that `std_b > std_a`.
# So when `a` and `b` are correlated, we prefer to drop `a`.

# Eaxmple 2: For pair (index_b, col_c), the std_mask is -1, meaning that `std_c < std_b`.
# So when `b` and `c` are correlated, we prefer to drop `c`.

# +
corr_mask * std_mask

# In this matrix, we combine two mask matrix together - dropping correlated featues with low std.
# The signature of matrix indicates the std_mask, and the absolute value of each indicates the correlation.
# For example, we first go through the column list fom a to f.
# The first pair we check would be (col_b, index_a) where the correlation is 0.995835.
# The number is larger than 0.9, meaning they are correlated, therefore one of them can be dropped.
# The number is positive, meaning std_b > std_a, therefore `a` should be dropped.
# -

missing_diff

missing_mask

# +
corr_mask * missing_mask

# In this matrix, we combine two mask matrix together - dropping correlated featues with more #missings.
# The signature of matrix indicates the std_mask, and the absolute value of each indicates the correlation.
# For example, we first go through the column list fom a to f.
# The first pair we check would be (col_b, index_a) where the abs(correlation) is 0.995835.
# The number is larger than 0.9, meaning they are correlated, therefore one of them can be dropped.
# The number is positive, meaning #missings_b < #missings_a, therefore `a` should be dropped.
# -

# #### 3.1 Smart feature selection based on `correlation` and `std`
# When two feature are correlated, drop the one with `lower standard deviation`.

feats_to_keep, feats_to_drop = make_selection(
    corr_mask=corr_mask, matrix_mask=std_mask, corr_thresh=0.8, second_matrix_type="std"
)

feats_to_keep, feats_to_drop

# Check if we still have correlated features dataframe:

assert data[data[feats_to_keep].corr().abs() > 0.8].sum().sum() == 0.0, "There are still correlated features"

# #### 3.2 Smart feature selection based on `correlation` and `missings`
# When two feature are correlated, drop the one with `less missing values`.

feats_to_keep, feats_to_drop = make_selection(
    corr_mask=corr_mask, matrix_mask=missing_mask, corr_thresh=0.8, second_matrix_type="missings"
)

feats_to_keep, feats_to_drop

# Check if we still have correlated features in the dataframe

assert data[data[feats_to_keep].corr().abs() > 0.8].sum().sum() == 0.0, "There are still correlated features"
