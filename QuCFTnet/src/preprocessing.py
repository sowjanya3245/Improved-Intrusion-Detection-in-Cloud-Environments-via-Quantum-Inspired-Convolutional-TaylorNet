import os
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_cic_ids2018(data_path):
    """Load and combine CIC-IDS2018 CSV files."""

    csv_files = sorted(
        glob.glob(
            os.path.join(data_path, "*.csv")
        )
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in: {data_path}"
        )

    dataframes = []

    for file in csv_files:
        print(f"Loading: {os.path.basename(file)}")

        df = pd.read_csv(
            file,
            low_memory=False
        )

        dataframes.append(df)

    data = pd.concat(
        dataframes,
        ignore_index=True
    )

    print(
        f"Loaded dataset shape: {data.shape}"
    )

    return data


def clean_data(df):
    """
    Basic cleaning of CIC-IDS2018.

    Only operations that do not learn statistical information
    from the complete dataset are performed here.
    """

    df = df.copy()

    # Clean column names
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    # Replace infinite values with NaN
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove completely empty columns
    df = df.dropna(
        axis=1,
        how="all"
    )

    # Find label column
    label_candidates = [
        "Label",
        "label",
        "LABEL"
    ]

    label_column = None

    for column in label_candidates:
        if column in df.columns:
            label_column = column
            break

    if label_column is None:
        raise ValueError(
            "Label column not found in CIC-IDS2018 dataset."
        )

    # Convert labels to binary
    labels = (
        df[label_column]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    labels = np.where(
        labels == "BENIGN",
        0,
        1
    ).astype(np.int32)

    # Remove label column
    features = df.drop(
        columns=[label_column]
    )

    # Keep only numeric features
    features = features.select_dtypes(
        include=[np.number]
    )

    # Replace infinite values
    features = features.replace(
        [np.inf, -np.inf],
        np.nan
    )

    print(
        f"Numeric feature shape: {features.shape}"
    )

    return features, labels


def split_data(
    X,
    y,
    random_state=42
):
    """
    Stratified 70:15:15 train/validation/test split.
    """

    # 70% training, 30% temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=random_state
    )

    # 15% validation, 15% testing
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=random_state
    )

    print("\nData Split:")
    print(
        f"Training   : {X_train.shape}"
    )
    print(
        f"Validation : {X_val.shape}"
    )
    print(
        f"Testing    : {X_test.shape}"
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


def remove_zero_variance_features(
    X_train,
    X_val,
    X_test
):
    """
    Remove zero-variance features.

    Variance is calculated ONLY from the training set
    to prevent data leakage.
    """

    variance = X_train.var(
        axis=0,
        skipna=True
    )

    valid_columns = variance[
        variance > 0
    ].index

    X_train = X_train[
        valid_columns
    ]

    X_val = X_val[
        valid_columns
    ]

    X_test = X_test[
        valid_columns
    ]

    print(
        f"Features after zero-variance removal: "
        f"{len(valid_columns)}"
    )

    return (
        X_train,
        X_val,
        X_test
    )


def impute_missing_values(
    X_train,
    X_val,
    X_test
):
    """
    Replace missing values using training-set medians.

    The median values are calculated ONLY from the training
    data and then applied to validation and test data.
    """

    # Calculate medians only from training data
    train_medians = X_train.median()

    # Apply the training medians
    X_train = X_train.fillna(
        train_medians
    )

    X_val = X_val.fillna(
        train_medians
    )

    X_test = X_test.fillna(
        train_medians
    )

    # Handle columns whose training median is still NaN
    # because the entire training portion was missing.
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    X_test = X_test.fillna(0)

    return (
        X_train,
        X_val,
        X_test
    )


def normalize_data(
    X_train,
    X_val,
    X_test
):
    """
    Min-Max normalization.

    The scaler is fitted ONLY on the training data.
    """

    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_val_scaled = scaler.transform(
        X_val
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler
    )


def preprocess_dataset(
    data_path,
    random_state=42
):


    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    df = load_cic_ids2018(
        data_path
    )

    # --------------------------------------------------------
    # 2. Basic cleaning
    # --------------------------------------------------------

    X, y = clean_data(df)

    # --------------------------------------------------------
    # 3. Stratified 70:15:15 split
    # --------------------------------------------------------

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_data(
        X,
        y,
        random_state=random_state
    )

    # --------------------------------------------------------
    # 4. Remove zero-variance features
    #    Using training data ONLY
    # --------------------------------------------------------

    (
        X_train,
        X_val,
        X_test
    ) = remove_zero_variance_features(
        X_train,
        X_val,
        X_test
    )

    # --------------------------------------------------------
    # 5. Missing-value imputation
    #    Medians calculated from training data ONLY
    # --------------------------------------------------------

    (
        X_train,
        X_val,
        X_test
    ) = impute_missing_values(
        X_train,
        X_val,
        X_test
    )

    # --------------------------------------------------------
    # 6. Min-Max normalization
    #    Fitted on training data ONLY
    # --------------------------------------------------------

    (
        X_train,
        X_val,
        X_test,
        scaler
    ) = normalize_data(
        X_train,
        X_val,
        X_test
    )

    print("\nFinal Preprocessed Data:")
    print(
        f"Training   : {X_train.shape}"
    )
    print(
        f"Validation : {X_val.shape}"
    )
    print(
        f"Testing    : {X_test.shape}"
    )

    return {
        "X_train": X_train.astype(np.float32),
        "X_val": X_val.astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "y_train": y_train.astype(np.int32),
        "y_val": y_val.astype(np.int32),
        "y_test": y_test.astype(np.int32),
        "scaler": scaler
    }


if __name__ == "__main__":

    print(
        "QuCFTnet preprocessing module"
    )
