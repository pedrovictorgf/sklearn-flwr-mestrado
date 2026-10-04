import pandas as pd
import numpy as np

TRAINING_DATASET = "GenIDS-UNSW15"
TARGET_COLUMN = "binary"
datasets = {"GenIDS-UNSW15": pd.read_csv("sklearnexample/GenIDS-UNSW15.csv").drop(columns=['multiclass'])}

METADATA_COLUMNS = {
    "binary",
    "date",
    "hours",
    "expiration_id",
    "label",
    "mapped_label",
    "Timestamp",
    "src_ip",
    "src_mac",
    "src_oui",
    "dst_ip",
    "dst_mac",
    "dst_oui",
    "ip_version",
    "vlan_id",
    "tunnel_id",
}
CATEGORICAL_COLUMNS = ["application_name", "application_category_name"]


def encode_shared_categories(dataframes, columns):
    encoded = {name: frame.copy() for name, frame in dataframes.items()}
    for column in columns:
        available = [frame[column].astype(str) for frame in encoded.values() if column in frame]
        if not available:
            continue
        categories = sorted(pd.concat(available, ignore_index=True).unique())
        mapping = {value: index for index, value in enumerate(categories)}
        for frame in encoded.values():
            if column in frame:
                frame[column] = frame[column].astype(str).map(mapping).astype(np.int64)
    return encoded


def encode_target(dataframes, target_column):
    encoded = {name: frame.copy() for name, frame in dataframes.items()}
    values = pd.concat(
        [frame[target_column] for frame in encoded.values()], ignore_index=True
    )
    if pd.api.types.is_numeric_dtype(values):
        classes = sorted(values.dropna().unique().tolist())
        mapping = {value: int(value) for value in classes}
    else:
        classes = sorted(values.dropna().astype(str).unique().tolist())
        mapping = {value: index for index, value in enumerate(classes)}
        for frame in encoded.values():
            frame[target_column] = frame[target_column].astype(str)
    for frame in encoded.values():
        frame[target_column] = frame[target_column].map(mapping).astype(np.int64)
    return encoded, mapping


def prepare_datasets(dataframes, target_column):
    normalized = {name: frame.copy() for name, frame in dataframes.items()}
    for frame in normalized.values():
        if target_column not in frame and "label" in frame:
            frame[target_column] = frame["label"]
    prepared, target_mapping = encode_target(normalized, target_column)
    prepared = encode_shared_categories(prepared, CATEGORICAL_COLUMNS)

    features = {}
    targets = {}
    reference_columns = None
    for name, frame in prepared.items():
        drop_columns = [
            column for column in METADATA_COLUMNS
            if column != target_column and column in frame.columns
        ]
        frame = frame.drop(columns=drop_columns)
        targets[name] = frame.pop(target_column).astype(np.int64)
        frame = frame.apply(pd.to_numeric, errors="raise").astype(np.float64)
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError(f"{name} contains NaN or infinite feature values.")
        if reference_columns is None:
            reference_columns = frame.columns.tolist()
        elif set(frame.columns) != set(reference_columns):
            missing = sorted(set(reference_columns) - set(frame.columns))
            extra = sorted(set(frame.columns) - set(reference_columns))
            raise ValueError(
                f"Feature mismatch in {name}. Missing: {missing}; extra: {extra}"
            )
        features[name] = frame.loc[:, reference_columns]

    features[TRAINING_DATASET][TARGET_COLUMN] = targets[TRAINING_DATASET]

    return features[TRAINING_DATASET]
# for col in df.columns:
#     if col == "binary" or col == "multiclass":
#         continue
#     df[col] = pd.to_numeric(df[col], errors='coerce')


# print("DTypes finais:\n", df.dtypes.value_counts())

# y = df['binary'].astype(str)  # string para o label encoder
# X = df.drop(columns=['binary'])

# numeric_cols     = X.select_dtypes(include=[np.number]).columns.tolist()
# categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
#
# print(f"{len(numeric_cols)} colunas numéricas e {len(categorical_cols)} colunas categóricas")

# features, targets, target_mapping = prepare_datasets(datasets, TARGET_COLUMN)

df = prepare_datasets(datasets, TARGET_COLUMN)
# pd.concat