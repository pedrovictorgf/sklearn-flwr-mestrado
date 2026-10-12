"""Prepare dataset CSV files and split them into the parquet partitions of the clients.

Usage:
    python -m sklearnexample.preprocess \\
        --dataset sklearnexample/GenIDS-UNSW15.csv 2 \\
        --dataset sklearnexample/GenIDS-CIC17.csv 2

Each --dataset takes a CSV file and the number of partitions to create from it.
The metadata columns are dropped, the categorical columns and the target are
encoded with a mapping shared by all the datasets, and the partitions are written
as client_<id>.parquet, numbered across all datasets in the order given, together
with a partitions.json describing each of them.

The CSV files are never loaded whole. A first pass reads only the target and the
categorical columns; a second pass streams each file in blocks, prepares each block
and appends its rows to the parquet file of their partition.
"""

import argparse
import json
import sys
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv
import pyarrow.parquet as pq
from pathlib import Path
from datasets import Dataset
from flwr_datasets.partitioner import DirichletPartitioner

TARGET_COLUMN = "binary"

OUTPUT_DIR = Path("sklearnexample/partitions")

METADATA_COLUMNS = {
    "binary",
    "multiclass",
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

# Size of the blocks in which the CSV files are streamed. The reader keeps several
# parsed blocks in memory at once, so this is what bounds the memory used.
CSV_BLOCK_SIZE = 4 * 1024 * 1024
# Rows gathered for a partition before they are written as one parquet row group
ROW_GROUP_SIZE = 100_000


def is_numeric(data_type):
    return pa.types.is_integer(data_type) or pa.types.is_floating(data_type)


def as_strings(column):
    """Return the column as strings, with the missing values as "nan"."""
    return pc.fill_null(pc.cast(column, pa.string()), "nan")


def scan_dataset(file_path, target_column):
    """First pass over a dataset CSV: read its header, target and categories.

    The file is streamed in blocks and only the target and the distinct values of
    the categorical columns are kept, so this stays small however large it is.
    """
    schema = pa.csv.open_csv(file_path).schema

    source_target = target_column
    if target_column not in schema.names:
        if "label" not in schema.names:
            raise ValueError(
                f"{file_path}: target column '{target_column}' not found. "
                f"Columns: {schema.names}"
            )
        source_target = "label"

    feature_names = [
        name for name in schema.names
        if name not in METADATA_COLUMNS and name != target_column
    ]
    # Categorical columns that are already numeric were encoded before
    categorical = [
        name for name in CATEGORICAL_COLUMNS
        if name in feature_names and not is_numeric(schema.field(name).type)
    ]
    non_numeric = [
        name for name in feature_names
        if name not in categorical and not is_numeric(schema.field(name).type)
    ]
    if non_numeric:
        raise ValueError(
            f"{file_path}: non-numeric feature columns {non_numeric}. Add them to "
            "METADATA_COLUMNS to drop them or to CATEGORICAL_COLUMNS to encode them."
        )

    reader = pa.csv.open_csv(
        file_path,
        read_options=pa.csv.ReadOptions(block_size=CSV_BLOCK_SIZE),
        convert_options=pa.csv.ConvertOptions(
            include_columns=[source_target] + categorical,
            column_types={name: pa.string() for name in categorical},
            strings_can_be_null=True,
        ),
    )
    target_chunks = []
    categories = {name: set() for name in categorical}
    for batch in reader:
        target_chunks.append(batch[source_target])
        for name in categorical:
            categories[name].update(pc.unique(as_strings(batch[name])).to_pylist())
    return {
        "file_path": file_path,
        "feature_names": feature_names,
        "target": pa.chunked_array(target_chunks),
        "categories": categories,
    }


def encode_shared_categories(scans, columns):
    """Map the values of each categorical column to an index shared by all datasets."""
    mappings = {}
    for column in columns:
        categories = set().union(*(scan["categories"].get(column, ()) for scan in scans))
        if categories:
            mappings[column] = pa.array(sorted(categories), type=pa.string())
    return mappings


def encode_target(scans):
    """Map the target values to integers, with a mapping shared by all datasets.

    A numeric target keeps its values; otherwise the classes are numbered in
    alphabetical order. Sets `target` of each scan to the encoded numpy array and
    returns the classes when they were numbered, or None.
    """
    classes = None
    if all(is_numeric(scan["target"].type) for scan in scans):
        for scan in scans:
            if scan["target"].null_count:
                raise ValueError(f"{scan['file_path']}: missing target values.")
            scan["target"] = scan["target"].to_numpy().astype(np.int64)
    else:
        values = [pc.cast(scan["target"], pa.string()) for scan in scans]
        classes = pa.array(
            sorted(set().union(*(pc.unique(column).drop_null().to_pylist()
                                 for column in values))),
            type=pa.string(),
        )
        for scan, column in zip(scans, values):
            if column.null_count:
                raise ValueError(f"{scan['file_path']}: missing target values.")
            scan["target"] = pc.index_in(column, value_set=classes).to_numpy().astype(np.int64)

    for scan in scans:
        if not np.isin(scan["target"], [0, 1]).all():
            raise ValueError(
                f"{scan['file_path']}: the target must have two classes, encoded as "
                f"0 and 1. Found: {np.unique(scan['target']).tolist()}"
                + (f" for classes {classes.to_pylist()}" if classes is not None else "")
            )
    return classes


def iter_prepared_batches(scan, feature_names, category_mappings, target_column):
    """Second pass over a dataset CSV: yield it prepared, one block at a time.

    Each batch has the features in float64, in the order of `feature_names`
    because the clients take them by position, followed by the target in int64.
    The metadata columns are not even parsed.
    """
    file_path = scan["file_path"]
    column_types = {
        name: pa.string() if name in scan["categories"] else pa.float64()
        for name in feature_names
    }
    reader = pa.csv.open_csv(
        file_path,
        read_options=pa.csv.ReadOptions(block_size=CSV_BLOCK_SIZE),
        convert_options=pa.csv.ConvertOptions(
            include_columns=feature_names,
            column_types=column_types,
            strings_can_be_null=True,
        ),
    )
    offset = 0
    for batch in reader:
        columns = []
        for name in feature_names:
            column = batch[name]
            if name in scan["categories"]:
                column = pc.cast(
                    pc.index_in(as_strings(column), value_set=category_mappings[name]),
                    pa.float64(),
                )
            if not np.isfinite(column.to_numpy(zero_copy_only=False)).all():
                raise ValueError(f"{file_path}: column '{name}' has NaN or infinite values.")
            columns.append(column)
        columns.append(pa.array(scan["target"][offset:offset + len(batch)]))
        offset += len(batch)
        yield pa.table(columns, names=feature_names + [target_column])

    if offset != len(scan["target"]):
        raise ValueError(f"{file_path}: changed while it was being read.")


def partition_indices(target, num_partitions, alpha, seed):
    """Return the row indices of each partition, split by the target with Dirichlet."""
    if num_partitions == 1:
        return [np.arange(len(target))]

    # The partitioner only needs the target, so the features stay out of it
    partitioner = DirichletPartitioner(num_partitions=num_partitions,
                                       partition_by="target",
                                       alpha=alpha,
                                       seed=seed)
    partitioner.dataset = Dataset.from_dict({"target": target,
                                             "row": np.arange(len(target))})
    return [
        np.asarray(partitioner.load_partition(partition_id)["row"])
        for partition_id in range(num_partitions)
    ]


def generate_partitions(datasets, output_dir=OUTPUT_DIR, target_column=TARGET_COLUMN,
                        alpha=0.5, seed=42, overwrite=False):
    """Prepare each (csv file, number of partitions) and write the parquet files."""
    output_dir = Path(output_dir)

    # Scan and check every dataset before touching the output directory
    scans = [scan_dataset(file_path, target_column) for file_path, _ in datasets]
    feature_names = scans[0]["feature_names"]
    for scan in scans[1:]:
        missing = sorted(set(feature_names) - set(scan["feature_names"]))
        extra = sorted(set(scan["feature_names"]) - set(feature_names))
        if missing or extra:
            raise ValueError(
                f"{scan['file_path']}: features differ from the first dataset. "
                f"Missing: {missing}; extra: {extra}"
            )
    category_mappings = encode_shared_categories(scans, CATEGORICAL_COLUMNS)
    target_classes = encode_target(scans)

    existing = sorted(output_dir.glob("client_*.parquet"))
    if existing and not overwrite:
        raise ValueError(
            f"{output_dir} already has {len(existing)} partitions; "
            "use --overwrite to replace them or another --output-dir"
        )
    for file in existing:
        file.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    schema = pa.schema(
        [(name, pa.float64()) for name in feature_names] + [(target_column, pa.int64())]
    )
    manifest = []
    for scan, (file_path, num_partitions) in zip(scans, datasets):
        target = scan["target"]

        # Partition of each row, and one parquet file per partition
        assignment = np.empty(len(target), dtype=np.int32)
        writers = []
        for local_id, indices in enumerate(
            partition_indices(target, num_partitions, alpha, seed)
        ):
            assignment[indices] = local_id
            output_file = output_dir / f"client_{len(manifest)}.parquet"
            writers.append(pq.ParquetWriter(output_file, schema))
            entry = {
                "file": output_file.name,
                "dataset": str(file_path),
                "partition": local_id,
                "rows": len(indices),
                "class_counts": {
                    str(value): int(count)
                    for value, count in zip(*np.unique(target[indices], return_counts=True))
                },
            }
            manifest.append(entry)
            print(f"{output_file}: {entry['rows']} rows, classes {entry['class_counts']}")

        # Rows of each partition waiting to be written
        pending = [[] for _ in writers]

        def flush(local_id):
            if pending[local_id]:
                writers[local_id].write_table(pa.concat_tables(pending[local_id]))
                pending[local_id].clear()

        try:
            offset = 0
            for batch in iter_prepared_batches(
                scan, feature_names, category_mappings, target_column
            ):
                batch_assignment = assignment[offset:offset + len(batch)]
                offset += len(batch)
                for local_id in range(num_partitions):
                    rows = batch.filter(pa.array(batch_assignment == local_id))
                    if len(rows):
                        pending[local_id].append(rows)
                    if sum(len(table) for table in pending[local_id]) >= ROW_GROUP_SIZE:
                        flush(local_id)
            for local_id in range(num_partitions):
                flush(local_id)
        finally:
            for writer in writers:
                writer.close()
        scan["target"] = None

    (output_dir / "partitions.json").write_text(json.dumps({
        "target_column": target_column,
        "target_classes": target_classes.to_pylist() if target_classes is not None else None,
        "categories": {name: values.to_pylist() for name, values in category_mappings.items()},
        "alpha": alpha,
        "seed": seed,
        "features": feature_names,
        "partitions": manifest,
    }, indent=2))


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

# df = prepare_datasets(datasets, TARGET_COLUMN)

# pd.concat


def main():
    parser = argparse.ArgumentParser(
        description="Split dataset CSV files into the parquet partitions of the clients."
    )
    parser.add_argument("--dataset", nargs=2, action="append", required=True,
                        metavar=("CSV", "PARTITIONS"),
                        help="CSV file and number of partitions to create from it "
                             "(repeat for each dataset)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, type=Path)
    parser.add_argument("--target", default=TARGET_COLUMN,
                        help="name of the 0/1 target column")
    parser.add_argument("--alpha", default=0.5, type=float,
                        help="Dirichlet concentration; lower is more non-IID")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--overwrite", action="store_true",
                        help="replace the partitions already in the output directory")
    args = parser.parse_args()

    datasets = []
    for file_path, num_partitions in args.dataset:
        if not Path(file_path).is_file():
            parser.error(f"dataset not found: {file_path}")
        if not num_partitions.isdigit() or int(num_partitions) < 1:
            parser.error(f"invalid number of partitions for {file_path}: {num_partitions}")
        datasets.append((file_path, int(num_partitions)))

    try:
        generate_partitions(datasets, args.output_dir, args.target, args.alpha,
                            args.seed, args.overwrite)
    except (ValueError, pa.ArrowInvalid) as error:
        sys.exit(f"error: {error}")


if __name__ == "__main__":
    main()
