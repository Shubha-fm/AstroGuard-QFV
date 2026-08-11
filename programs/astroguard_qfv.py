#!/usr/bin/env python3
"""
AstroGuard-QFV large-dataset pipeline.

Install:
    pip install alerce pennylane torch scikit-learn pandas numpy matplotlib tqdm

Run:
    python astroguard_qfv.py

The script:
- retrieves a balanced 60,000-object maximum-target ZTF/ALeRCE dataset;
- preprocesses light curves;
- trains a 4-layer Transformer + 4-qubit quantum feature model;
- trains a 5-member ensemble;
- evaluates classification and uncertainty;
- applies a formal release gate;
- saves models, predictions, plots, and result summaries.

The complete hybrid model runs on CPU because PennyLane default.qubit and
CUDA tensors can otherwise produce device-mismatch errors.
"""

import copy
import os
import inspect
import json
import math
import random
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

from alerce.core import Alerce
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42
TARGET_PER_CLASS = 20000
OVERSAMPLING_FACTOR = 1.60
QUERY_BATCH_SIZE = 500

MIN_OBSERVATIONS = 20
MAX_SEQUENCE_LENGTH = 128
INPUT_DIMENSION = 6

BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 8

MODEL_DIMENSION = 128
NUMBER_OF_HEADS = 4
TRANSFORMER_LAYERS = 4
FEEDFORWARD_DIMENSION = 256
DROPOUT = 0.15
PROJECTION_DIMENSION = 64

N_QUBITS = 4
QUANTUM_LAYERS = 2

RECONSTRUCTION_WEIGHT = 0.10
CONTRASTIVE_WEIGHT = 0.10
CLASSIFICATION_WEIGHT = 0.80
TEMPERATURE = 0.20

MASK_PROBABILITY = 0.20
POINT_DROPOUT = 0.10
JITTER_STANDARD_DEVIATION = 0.03

ENSEMBLE_SEEDS = [42, 142, 242, 342, 442]

CONFIDENCE_THRESHOLD = 0.50
UNCERTAINTY_THRESHOLD = 0.10
QUALITY_THRESHOLD = 0.30

OUTPUT_DIRECTORY = Path(os.environ.get("ASTROGUARD_OUTPUT", str(Path(__file__).resolve().parents[1] / "output")))
CACHE_DIRECTORY = OUTPUT_DIRECTORY / "cache"
LIGHT_CURVE_DIRECTORY = CACHE_DIRECTORY / "light_curves"
MODEL_DIRECTORY = OUTPUT_DIRECTORY / "models"
RESULT_DIRECTORY = OUTPUT_DIRECTORY / "results"
PLOT_DIRECTORY = OUTPUT_DIRECTORY / "plots"

CLASS_GROUPS = {
    "Transient": ["SN", "SNIa", "SNIbc", "SNII"],
    "Periodic": ["Periodic-Other", "RR Lyrae", "Eclipsing Binary", "LPV"],
    "Stochastic": ["AGN", "QSO", "Blazar", "YSO"],
}


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def prepare_environment():
    warnings.filterwarnings("ignore")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    for directory in [
        OUTPUT_DIRECTORY,
        CACHE_DIRECTORY,
        LIGHT_CURVE_DIRECTORY,
        MODEL_DIRECTORY,
        RESULT_DIRECTORY,
        PLOT_DIRECTORY,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def normalize_name(value):
    return (
        str(value)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def find_column(dataframe, alternatives):
    for column in alternatives:
        if column in dataframe.columns:
            return column
    return None


# =============================================================================
# ALERCE DATASET RETRIEVAL
# =============================================================================

def discover_classifier(client):
    """
    Discover the current ZTF light-curve classifier.

    ALeRCE Python Client 2.x documents:
        query_classifiers(format="pandas", survey="ztf")
        query_classes(classifier_name, classifier_version,
                      format="pandas", survey="ztf")

    We first try the well-known hierarchical random-forest version and then
    fall back to the classifier catalogue returned by the service.
    """
    preferred = [
        ("lc_classifier", "hierarchical_random_forest_1.0.0"),
    ]

    for classifier, version in preferred:
        for kwargs in (
            {"format": "pandas", "survey": "ztf"},
            {"format": "pandas"},
            {},
        ):
            try:
                result = client.query_classes(
                    classifier,
                    version,
                    **kwargs,
                )
                classes_df = pd.DataFrame(result)
                if not classes_df.empty:
                    return classifier, version, classes_df
            except Exception:
                pass

    try:
        classifiers = client.query_classifiers(
            format="pandas",
            survey="ztf",
        )
    except Exception:
        classifiers = client.query_classifiers(
            format="pandas"
        )

    classifiers_df = pd.DataFrame(classifiers)

    if classifiers_df.empty:
        raise RuntimeError(
            "ALeRCE returned an empty classifier catalogue."
        )

    name_col = find_column(
        classifiers_df,
        ["classifier_name", "name", "classifier"],
    )
    version_col = find_column(
        classifiers_df,
        ["classifier_version", "version"],
    )

    if name_col is None:
        raise RuntimeError(
            "Could not identify classifier-name column. "
            f"Columns: {classifiers_df.columns.tolist()}"
        )

    # Prefer the light-curve classifier.
    rows = classifiers_df[
        classifiers_df[name_col]
        .astype(str)
        .str.contains("lc_classifier", case=False, na=False)
    ]

    if rows.empty:
        rows = classifiers_df

    errors = []

    for _, row in rows.iterrows():
        classifier = str(row[name_col])
        version = (
            str(row[version_col])
            if version_col is not None
            and pd.notna(row[version_col])
            else "hierarchical_random_forest_1.0.0"
        )

        for kwargs in (
            {"format": "pandas", "survey": "ztf"},
            {"format": "pandas"},
            {},
        ):
            try:
                result = client.query_classes(
                    classifier,
                    version,
                    **kwargs,
                )
                classes_df = pd.DataFrame(result)
                if not classes_df.empty:
                    return classifier, version, classes_df
            except Exception as exc:
                errors.append(
                    f"{classifier}/{version}: {exc}"
                )

    raise RuntimeError(
        "No compatible ALeRCE light-curve classifier was discovered.\n"
        + "\n".join(errors[-5:])
    )

def match_available_classes(classes_df):
    class_column = find_column(
        classes_df,
        ["class_name", "name", "class", "classifier_class"],
    )

    if class_column is None:
        raise ValueError(
            "Class-name column not found. Available columns: "
            f"{classes_df.columns.tolist()}"
        )

    normalized_available = {
        normalize_name(name): str(name)
        for name in classes_df[class_column].astype(str)
    }

    matched_groups = {}

    for broad_class, requested_classes in CLASS_GROUPS.items():
        matched = []

        for requested in requested_classes:
            normalized_requested = normalize_name(requested)

            if normalized_requested in normalized_available:
                matched.append(
                    normalized_available[normalized_requested]
                )

        if matched:
            matched_groups[broad_class] = matched

    if len(matched_groups) < 3:
        classes_df.to_csv(
            RESULT_DIRECTORY / "available_classes.csv",
            index=False,
        )
        raise ValueError(
            "All three broad groups could not be matched. "
            "Inspect available_classes.csv and update CLASS_GROUPS."
        )

    return matched_groups


def make_object_query(client, classifier, version):
    """
    Return a page-query function for the current ALeRCE ZTF API.

    classifier_version is used when querying the taxonomy, but query_objects
    filters ZTF objects by classifier name + class_name.
    """
    def query_page(class_name, page_number, page_size):
        kwargs = {
            "survey": "ztf",
            "format": "pandas",
            "classifier": classifier,
            "class_name": class_name,
            "page": int(page_number),
            "page_size": int(page_size),
            "order_by": "probability",
            "order_mode": "DESC",
        }

        try:
            result = client.query_objects(**kwargs)
        except TypeError:
            # Compatibility with older ALeRCE clients that do not expose survey.
            kwargs.pop("survey", None)
            result = client.query_objects(**kwargs)

        return pd.DataFrame(result)

    return query_page

def detect_oid_column(dataframe):
    column = find_column(
        dataframe,
        ["oid", "objectId", "object_id", "id"],
    )

    if column is None:
        raise ValueError(
            "Object-ID column not found. Available columns: "
            f"{dataframe.columns.tolist()}"
        )

    return column


def collect_ids(
    query_page,
    class_name,
    requested_count,
    oid_column,
):
    collected = []
    seen = set()
    page_number = 1
    maximum_pages = (
        int(np.ceil(requested_count / QUERY_BATCH_SIZE))
        + 15
    )

    while (
        len(collected) < requested_count
        and page_number <= maximum_pages
    ):
        try:
            page_df = query_page(
                class_name,
                page_number,
                QUERY_BATCH_SIZE,
            )
        except Exception as error:
            print(
                f"Query error for {class_name}, "
                f"page {page_number}: {error}"
            )
            time.sleep(3)
            page_number += 1
            continue

        if page_df.empty:
            break

        if oid_column not in page_df.columns:
            oid_column = detect_oid_column(page_df)

        page_ids = (
            page_df[oid_column]
            .dropna()
            .astype(str)
            .tolist()
        )

        for oid in page_ids:
            if oid not in seen:
                collected.append(oid)
                seen.add(oid)

        print(
            f"{class_name}: "
            f"{len(collected)}/{requested_count}"
        )

        if len(page_ids) < QUERY_BATCH_SIZE:
            break

        page_number += 1
        time.sleep(0.2)

    return collected[:requested_count]


def build_candidate_table(client):
    classifier, version, classes_df = (
        discover_classifier(client)
    )

    classes_df.to_csv(
        RESULT_DIRECTORY / "available_classes.csv",
        index=False,
    )

    print("Classifier:", classifier)
    print("Version:", version)

    matched_groups = match_available_classes(
        classes_df
    )

    with open(
        RESULT_DIRECTORY / "matched_class_groups.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(matched_groups, file, indent=2)

    query_page = make_object_query(
        client,
        classifier,
        version,
    )

    first_class = next(
        iter(matched_groups.values())
    )[0]

    test_df = query_page(
        first_class,
        1,
        10,
    )

    if test_df.empty:
        raise RuntimeError(
            "The ALeRCE test query returned no objects."
        )

    oid_column = detect_oid_column(test_df)

    requested_per_broad_class = int(
        TARGET_PER_CLASS * OVERSAMPLING_FACTOR
    )

    rows = []

    for broad_class, alerce_classes in (
        matched_groups.items()
    ):
        per_alerce_class = int(
            np.ceil(
                requested_per_broad_class
                / len(alerce_classes)
            )
        )

        broad_ids = set()

        for alerce_class in alerce_classes:
            class_ids = collect_ids(
                query_page,
                alerce_class,
                per_alerce_class,
                oid_column,
            )

            for oid in class_ids:
                if oid in broad_ids:
                    continue

                rows.append(
                    {
                        "oid": oid,
                        "label": broad_class,
                        "alerce_class": alerce_class,
                    }
                )
                broad_ids.add(oid)

            if len(broad_ids) >= requested_per_broad_class:
                break

        print(
            f"{broad_class}: "
            f"{len(broad_ids)} candidate IDs"
        )

    objects_df = pd.DataFrame(rows)

    if objects_df.empty:
        raise RuntimeError(
            "No candidate objects were collected."
        )

    label_count = (
        objects_df.groupby("oid")["label"].nunique()
    )

    conflicting_oids = set(
        label_count[label_count > 1].index
    )

    objects_df = objects_df[
        ~objects_df["oid"].isin(conflicting_oids)
    ]

    objects_df = (
        objects_df
        .drop_duplicates(subset="oid")
        .reset_index(drop=True)
    )

    objects_df.to_csv(
        RESULT_DIRECTORY / "candidate_objects.csv",
        index=False,
    )

    print(objects_df["label"].value_counts())

    return objects_df


# =============================================================================
# LIGHT-CURVE PREPROCESSING
# =============================================================================

def load_light_curve(client, oid):
    cache_path = LIGHT_CURVE_DIRECTORY / f"{oid}.pkl"

    if cache_path.exists():
        try:
            return pd.read_pickle(cache_path)
        except Exception:
            cache_path.unlink(missing_ok=True)

    try:
        try:
            detections = client.query_detections(
                oid=oid,
                survey="ztf",
                format="pandas",
            )
        except TypeError:
            detections = client.query_detections(
                oid=oid,
                format="pandas",
            )

        dataframe = pd.DataFrame(detections)

        if dataframe.empty:
            return None

        dataframe.to_pickle(cache_path)
        return dataframe

    except Exception:
        return None


def preprocess_light_curve(client, oid):
    light_curve = load_light_curve(client, oid)

    if light_curve is None or light_curve.empty:
        return None

    time_column = find_column(
        light_curve,
        ["mjd", "jd"],
    )
    magnitude_column = find_column(
        light_curve,
        ["magpsf", "mag"],
    )
    error_column = find_column(
        light_curve,
        ["sigmapsf", "magerr"],
    )
    filter_column = find_column(
        light_curve,
        ["fid", "filter"],
    )

    if (
        time_column is None
        or magnitude_column is None
        or error_column is None
    ):
        return None

    working = light_curve.copy()

    working["time_value"] = pd.to_numeric(
        working[time_column],
        errors="coerce",
    )
    working["magnitude_value"] = pd.to_numeric(
        working[magnitude_column],
        errors="coerce",
    )
    working["error_value"] = pd.to_numeric(
        working[error_column],
        errors="coerce",
    )

    working = working.dropna(
        subset=[
            "time_value",
            "magnitude_value",
            "error_value",
        ]
    )

    working = working[
        working["error_value"] > 0
    ].sort_values("time_value")

    if len(working) < MIN_OBSERVATIONS:
        return None

    if len(working) > MAX_SEQUENCE_LENGTH:
        positions = np.linspace(
            0,
            len(working) - 1,
            MAX_SEQUENCE_LENGTH,
        ).astype(int)

        working = working.iloc[positions].copy()

    times = working["time_value"].to_numpy(
        dtype=np.float32
    )
    magnitudes = working[
        "magnitude_value"
    ].to_numpy(dtype=np.float32)
    errors = working["error_value"].to_numpy(
        dtype=np.float32
    )

    normalized_time = (
        times - times.min()
    ) / max(times.max() - times.min(), 1e-6)

    median_magnitude = np.median(magnitudes)
    mad = np.median(
        np.abs(magnitudes - median_magnitude)
    )

    magnitude_scale = max(
        1.4826 * mad,
        np.std(magnitudes),
        1e-6,
    )

    normalized_magnitude = (
        magnitudes - median_magnitude
    ) / magnitude_scale

    normalized_error = np.clip(
        errors / max(np.median(errors), 1e-6),
        0,
        10,
    )

    g_filter = np.zeros(
        len(working),
        dtype=np.float32,
    )
    r_filter = np.zeros(
        len(working),
        dtype=np.float32,
    )

    if filter_column is not None:
        filters = (
            working[filter_column]
            .astype(str)
            .str.lower()
        )

        g_filter = filters.isin(
            ["1", "g", "ztf_g", "zg"]
        ).astype(np.float32)

        r_filter = filters.isin(
            ["2", "r", "ztf_r", "zr"]
        ).astype(np.float32)

    observed = np.ones(
        len(working),
        dtype=np.float32,
    )

    sequence = np.column_stack(
        [
            normalized_time,
            normalized_magnitude,
            normalized_error,
            g_filter,
            r_filter,
            observed,
        ]
    ).astype(np.float32)

    observation_score = min(
        len(working) / MAX_SEQUENCE_LENGTH,
        1.0,
    )

    uncertainty_score = float(
        np.exp(-np.median(errors))
    )

    filter_score = float(
        (g_filter.sum() > 0)
        and (r_filter.sum() > 0)
    )

    quality_score = float(
        np.clip(
            0.50 * observation_score
            + 0.35 * uncertainty_score
            + 0.15 * filter_score,
            0,
            1,
        )
    )

    return {
        "sequence": sequence,
        "length": len(sequence),
        "quality_score": quality_score,
    }


def build_processed_records(client, objects_df):
    processed_records = []
    class_names = sorted(
        objects_df["label"].unique()
    )

    successful_counts = {
        name: 0
        for name in class_names
    }

    shuffled = objects_df.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    progress = tqdm(
        total=TARGET_PER_CLASS * len(class_names),
        desc="Processing light curves",
    )

    for row in shuffled.itertuples(index=False):
        if (
            successful_counts[row.label]
            >= TARGET_PER_CLASS
        ):
            continue

        processed = preprocess_light_curve(
            client,
            row.oid,
        )

        if processed is None:
            continue

        processed_records.append(
            {
                "oid": row.oid,
                "label": row.label,
                "alerce_class": row.alerce_class,
                "sequence": processed["sequence"],
                "length": processed["length"],
                "quality_score": processed[
                    "quality_score"
                ],
            }
        )

        successful_counts[row.label] += 1
        progress.update(1)

        if all(
            count >= TARGET_PER_CLASS
            for count in successful_counts.values()
        ):
            break

    progress.close()

    balanced = []

    for class_name in class_names:
        class_records = [
            record
            for record in processed_records
            if record["label"] == class_name
        ]

        random.Random(SEED).shuffle(
            class_records
        )

        balanced.extend(
            class_records[:TARGET_PER_CLASS]
        )

    random.Random(SEED).shuffle(balanced)

    print("Successful counts:", successful_counts)
    print("Final dataset size:", len(balanced))

    summary_df = pd.DataFrame(
        {
            "oid": [r["oid"] for r in balanced],
            "label": [r["label"] for r in balanced],
            "alerce_class": [
                r["alerce_class"]
                for r in balanced
            ],
            "sequence_length": [
                r["length"]
                for r in balanced
            ],
            "quality_score": [
                r["quality_score"]
                for r in balanced
            ],
        }
    )

    summary_df.to_csv(
        RESULT_DIRECTORY / "dataset_summary.csv",
        index=False,
    )

    np.save(
        CACHE_DIRECTORY / f"processed_records_{TARGET_PER_CLASS}_per_class.npy",
        np.asarray(balanced, dtype=object),
        allow_pickle=True,
    )

    print(summary_df["label"].value_counts())

    return balanced


# =============================================================================
# DATASET AND DATALOADER
# =============================================================================

class LightCurveDataset(Dataset):
    def __init__(self, records, indices, labels):
        self.records = records
        self.indices = np.asarray(indices)
        self.labels = labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        record_index = int(
            self.indices[position]
        )
        record = self.records[record_index]

        return {
            "oid": record["oid"],
            "sequence": torch.tensor(
                record["sequence"],
                dtype=torch.float32,
            ),
            "label": torch.tensor(
                self.labels[record_index],
                dtype=torch.long,
            ),
            "quality": torch.tensor(
                record["quality_score"],
                dtype=torch.float32,
            ),
        }


def collate_light_curves(batch):
    maximum_length = min(
        max(
            item["sequence"].shape[0]
            for item in batch
        ),
        MAX_SEQUENCE_LENGTH,
    )

    batch_size = len(batch)

    sequences = torch.zeros(
        batch_size,
        maximum_length,
        INPUT_DIMENSION,
        dtype=torch.float32,
    )

    padding_mask = torch.ones(
        batch_size,
        maximum_length,
        dtype=torch.bool,
    )

    labels = torch.empty(
        batch_size,
        dtype=torch.long,
    )

    quality_scores = torch.empty(
        batch_size,
        dtype=torch.float32,
    )

    object_ids = []

    for index, item in enumerate(batch):
        sequence = item["sequence"][
            -maximum_length:
        ]

        length = sequence.shape[0]

        sequences[index, :length] = sequence
        padding_mask[index, :length] = False
        labels[index] = item["label"]
        quality_scores[index] = item["quality"]
        object_ids.append(item["oid"])

    return {
        "oids": object_ids,
        "sequence": sequences,
        "padding_mask": padding_mask,
        "label": labels,
        "quality": quality_scores,
    }


def create_splits_and_loaders(records):
    label_encoder = LabelEncoder()

    encoded_labels = label_encoder.fit_transform(
        [record["label"] for record in records]
    )

    indices = np.arange(len(records))

    train_indices, temporary_indices = (
        train_test_split(
            indices,
            test_size=0.30,
            random_state=SEED,
            stratify=encoded_labels,
        )
    )

    validation_indices, test_indices = (
        train_test_split(
            temporary_indices,
            test_size=0.50,
            random_state=SEED,
            stratify=encoded_labels[
                temporary_indices
            ],
        )
    )

    train_dataset = LightCurveDataset(
        records,
        train_indices,
        encoded_labels,
    )

    validation_dataset = LightCurveDataset(
        records,
        validation_indices,
        encoded_labels,
    )

    test_dataset = LightCurveDataset(
        records,
        test_indices,
        encoded_labels,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_light_curves,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_light_curves,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_light_curves,
    )

    return (
        label_encoder,
        encoded_labels,
        train_indices,
        validation_indices,
        test_indices,
        train_loader,
        validation_loader,
        test_loader,
    )


# =============================================================================
# MODEL
# =============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()

        positions = torch.arange(
            MAX_SEQUENCE_LENGTH
        ).unsqueeze(1)

        divisors = torch.exp(
            torch.arange(
                0,
                MODEL_DIMENSION,
                2,
            )
            * (
                -math.log(10000.0)
                / MODEL_DIMENSION
            )
        )

        encoding = torch.zeros(
            MAX_SEQUENCE_LENGTH,
            MODEL_DIMENSION,
        )

        encoding[:, 0::2] = torch.sin(
            positions * divisors
        )
        encoding[:, 1::2] = torch.cos(
            positions * divisors
        )

        self.register_buffer(
            "encoding",
            encoding.unsqueeze(0),
        )

    def forward(self, tensor):
        return (
            tensor
            + self.encoding[:, : tensor.shape[1]]
        )


class HybridQuantumTransformer(nn.Module):
    def __init__(self, number_of_classes):
        super().__init__()

        self.input_projection = nn.Linear(
            INPUT_DIMENSION,
            MODEL_DIMENSION,
        )

        self.positional_encoding = PositionalEncoding()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=MODEL_DIMENSION,
            nhead=NUMBER_OF_HEADS,
            dim_feedforward=FEEDFORWARD_DIMENSION,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=TRANSFORMER_LAYERS,
            norm=nn.LayerNorm(MODEL_DIMENSION),
        )

        self.reconstruction_decoder = nn.Sequential(
            nn.Linear(
                MODEL_DIMENSION,
                FEEDFORWARD_DIMENSION,
            ),
            nn.GELU(),
            nn.Linear(
                FEEDFORWARD_DIMENSION,
                INPUT_DIMENSION,
            ),
        )

        self.contrastive_projection = nn.Sequential(
            nn.Linear(
                MODEL_DIMENSION,
                MODEL_DIMENSION,
            ),
            nn.GELU(),
            nn.Linear(
                MODEL_DIMENSION,
                PROJECTION_DIMENSION,
            ),
        )

        self.quantum_input = nn.Sequential(
            nn.LayerNorm(MODEL_DIMENSION),
            nn.Linear(
                MODEL_DIMENSION,
                N_QUBITS,
            ),
            nn.Tanh(),
        )

        quantum_device = qml.device(
            "default.qubit",
            wires=N_QUBITS,
        )

        @qml.qnode(
            quantum_device,
            interface="torch",
            diff_method="backprop",
        )
        def quantum_circuit(
            inputs,
            quantum_weights,
        ):
            qml.AngleEmbedding(
                inputs,
                wires=range(N_QUBITS),
                rotation="Y",
            )

            qml.StronglyEntanglingLayers(
                quantum_weights,
                wires=range(N_QUBITS),
            )

            return [
                qml.expval(qml.PauliZ(wire))
                for wire in range(N_QUBITS)
            ]

        self.quantum_layer = qml.qnn.TorchLayer(
            quantum_circuit,
            {
                "quantum_weights": (
                    QUANTUM_LAYERS,
                    N_QUBITS,
                    3,
                )
            },
        )

        fusion_dimension = (
            MODEL_DIMENSION + N_QUBITS
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dimension),
            nn.Linear(fusion_dimension, 128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, number_of_classes),
        )

    @staticmethod
    def masked_average(
        hidden_states,
        padding_mask,
    ):
        valid = (
            ~padding_mask
        ).unsqueeze(-1).float()

        denominator = (
            valid.sum(dim=1)
            .clamp_min(1.0)
        )

        return (
            hidden_states * valid
        ).sum(dim=1) / denominator

    def forward(
        self,
        sequence,
        padding_mask,
    ):
        hidden = self.input_projection(sequence)
        hidden = self.positional_encoding(hidden)

        hidden = self.transformer(
            hidden,
            src_key_padding_mask=padding_mask,
        )

        pooled = self.masked_average(
            hidden,
            padding_mask,
        )

        reconstruction = (
            self.reconstruction_decoder(hidden)
        )

        contrastive = F.normalize(
            self.contrastive_projection(pooled),
            dim=-1,
        )

        quantum_angles = (
            self.quantum_input(pooled)
            * math.pi
        )

        quantum_features = self.quantum_layer(
            quantum_angles
        ).float()

        fused = torch.cat(
            [pooled, quantum_features],
            dim=1,
        )

        logits = self.classifier(fused)

        return {
            "reconstruction": reconstruction,
            "contrastive_vector": contrastive,
            "quantum_features": quantum_features,
            "embedding": pooled,
            "logits": logits,
        }


# =============================================================================
# TRAINING
# =============================================================================

def augment_sequence(sequence, padding_mask):
    augmented = sequence.clone()
    valid = ~padding_mask

    noise = (
        torch.randn_like(
            augmented[:, :, :3]
        )
        * JITTER_STANDARD_DEVIATION
    )

    augmented[:, :, :3] += (
        noise * valid.unsqueeze(-1)
    )

    dropped = (
        torch.rand_like(valid.float())
        < POINT_DROPOUT
    ) & valid

    augmented[dropped] = 0.0

    feature_mask = (
        torch.rand_like(augmented)
        < MASK_PROBABILITY
    ) & valid.unsqueeze(-1)

    return (
        augmented.masked_fill(
            feature_mask,
            0.0,
        ),
        feature_mask,
    )


def reconstruction_loss(
    prediction,
    target,
    feature_mask,
    padding_mask,
):
    valid = (
        feature_mask
        & (~padding_mask).unsqueeze(-1)
    )

    if not valid.any():
        valid = (
            (~padding_mask)
            .unsqueeze(-1)
            .expand_as(target)
        )

    return (
        (prediction - target).pow(2)[valid]
        .mean()
    )


def contrastive_loss(first, second):
    batch_size = first.shape[0]

    representations = torch.cat(
        [first, second],
        dim=0,
    )

    similarities = (
        representations
        @ representations.T
    ) / TEMPERATURE

    identity = torch.eye(
        2 * batch_size,
        dtype=torch.bool,
        device=representations.device,
    )

    similarities = similarities.masked_fill(
        identity,
        -1e9,
    )

    targets = torch.arange(
        2 * batch_size,
        device=representations.device,
    )

    targets = (
        targets + batch_size
    ) % (2 * batch_size)

    return F.cross_entropy(
        similarities,
        targets,
    )


def make_classification_loss(
    encoded_labels,
    train_indices,
    number_of_classes,
    device,
):
    training_labels = encoded_labels[
        train_indices
    ]

    counts = np.bincount(
        training_labels,
        minlength=number_of_classes,
    )

    weights = (
        len(training_labels)
        / (
            number_of_classes
            * np.maximum(counts, 1)
        )
    )

    return nn.CrossEntropyLoss(
        weight=torch.tensor(
            weights,
            dtype=torch.float32,
            device=device,
        ),
        label_smoothing=0.03,
    )


def combined_loss(
    model,
    sequence,
    padding_mask,
    labels,
    classification_loss_function,
):
    first_view, first_mask = augment_sequence(
        sequence,
        padding_mask,
    )

    second_view, second_mask = augment_sequence(
        sequence,
        padding_mask,
    )

    first_output = model(
        first_view,
        padding_mask,
    )

    second_output = model(
        second_view,
        padding_mask,
    )

    reconstruction = (
        reconstruction_loss(
            first_output["reconstruction"],
            sequence,
            first_mask,
            padding_mask,
        )
        + reconstruction_loss(
            second_output["reconstruction"],
            sequence,
            second_mask,
            padding_mask,
        )
    ) / 2.0

    contrastive = contrastive_loss(
        first_output["contrastive_vector"],
        second_output["contrastive_vector"],
    )

    classification = (
        classification_loss_function(
            first_output["logits"],
            labels,
        )
        + classification_loss_function(
            second_output["logits"],
            labels,
        )
    ) / 2.0

    total = (
        RECONSTRUCTION_WEIGHT * reconstruction
        + CONTRASTIVE_WEIGHT * contrastive
        + CLASSIFICATION_WEIGHT * classification
    )

    return total


def evaluate(
    model,
    data_loader,
    classification_loss_function,
    device,
):
    model.eval()

    losses = []
    predictions = []
    labels_all = []
    probabilities_all = []

    with torch.no_grad():
        for batch in data_loader:
            sequence = batch["sequence"].to(device)
            padding_mask = batch[
                "padding_mask"
            ].to(device)
            labels = batch["label"].to(device)

            output = model(
                sequence,
                padding_mask,
            )

            logits = output["logits"]

            losses.append(
                classification_loss_function(
                    logits,
                    labels,
                ).item()
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predictions.extend(
                probabilities.argmax(dim=1)
                .cpu()
                .numpy()
            )

            labels_all.extend(
                labels.cpu().numpy()
            )

            probabilities_all.extend(
                probabilities.cpu().numpy()
            )

    predictions = np.asarray(predictions)
    labels_all = np.asarray(labels_all)

    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(
            labels_all,
            predictions,
        ),
        "macro_f1": f1_score(
            labels_all,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "predictions": predictions,
        "labels": labels_all,
        "probabilities": np.asarray(
            probabilities_all
        ),
    }


def train_model(
    model,
    train_loader,
    validation_loader,
    classification_loss_function,
    device,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS,
        )
    )

    best_f1 = -1.0
    best_state = None
    waiting = 0
    history = []

    for epoch in range(EPOCHS):
        model.train()
        training_losses = []

        for batch in train_loader:
            sequence = batch["sequence"].to(
                device
            )
            padding_mask = batch[
                "padding_mask"
            ].to(device)
            labels = batch["label"].to(device)

            loss = combined_loss(
                model,
                sequence,
                padding_mask,
                labels,
                classification_loss_function,
            )

            optimizer.zero_grad(
                set_to_none=True
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()
            training_losses.append(loss.item())

        scheduler.step()

        validation = evaluate(
            model,
            validation_loader,
            classification_loss_function,
            device,
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(
                    np.mean(training_losses)
                ),
                "validation_loss": validation[
                    "loss"
                ],
                "validation_accuracy": validation[
                    "accuracy"
                ],
                "validation_macro_f1": validation[
                    "macro_f1"
                ],
            }
        )

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train {history[-1]['train_loss']:.4f} | "
            f"Val acc {validation['accuracy'] * 100:.2f}% | "
            f"Val F1 {validation['macro_f1'] * 100:.2f}%"
        )

        if validation["macro_f1"] > best_f1:
            best_f1 = validation["macro_f1"]
            best_state = copy.deepcopy(
                model.state_dict()
            )
            waiting = 0
        else:
            waiting += 1

        if waiting >= PATIENCE:
            print("Early stopping.")
            break

    if best_state is None:
        raise RuntimeError(
            "No model checkpoint was created."
        )

    model.load_state_dict(best_state)

    return model, pd.DataFrame(history)


def predict_model(model, data_loader, device):
    model.eval()

    probability_batches = []
    label_batches = []
    quality_batches = []
    object_ids = []

    with torch.no_grad():
        for batch in data_loader:
            sequence = batch["sequence"].to(
                device
            )
            padding_mask = batch[
                "padding_mask"
            ].to(device)

            output = model(
                sequence,
                padding_mask,
            )

            probability_batches.append(
                torch.softmax(
                    output["logits"],
                    dim=1,
                )
                .cpu()
                .numpy()
            )

            label_batches.append(
                batch["label"].numpy()
            )

            quality_batches.append(
                batch["quality"].numpy()
            )

            object_ids.extend(batch["oids"])

    return {
        "probabilities": np.concatenate(
            probability_batches,
            axis=0,
        ),
        "labels": np.concatenate(
            label_batches,
            axis=0,
        ),
        "quality": np.concatenate(
            quality_batches,
            axis=0,
        ),
        "oids": np.asarray(object_ids),
    }



# =============================================================================
# CALIBRATION, SELECTIVE PREDICTION, AND REPORTING
# =============================================================================

def expected_calibration_error(
    true_labels,
    predictions,
    confidence,
    n_bins=15,
):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    reliability_rows = []

    for i in range(n_bins):
        left = edges[i]
        right = edges[i + 1]

        if i == n_bins - 1:
            mask = (
                (confidence >= left)
                & (confidence <= right)
            )
        else:
            mask = (
                (confidence >= left)
                & (confidence < right)
            )

        count = int(mask.sum())
        if count == 0:
            reliability_rows.append(
                {
                    "bin_left": left,
                    "bin_right": right,
                    "count": 0,
                    "mean_confidence": np.nan,
                    "accuracy": np.nan,
                }
            )
            continue

        bin_conf = float(confidence[mask].mean())
        bin_acc = float(
            (predictions[mask] == true_labels[mask]).mean()
        )

        ece += (
            count / len(true_labels)
        ) * abs(bin_acc - bin_conf)

        reliability_rows.append(
            {
                "bin_left": left,
                "bin_right": right,
                "count": count,
                "mean_confidence": bin_conf,
                "accuracy": bin_acc,
            }
        )

    return float(ece), pd.DataFrame(reliability_rows)


def uncertainty_error_auroc(
    true_labels,
    predictions,
    uncertainty,
):
    error_labels = (
        np.asarray(true_labels)
        != np.asarray(predictions)
    ).astype(int)

    if len(np.unique(error_labels)) < 2:
        return float("nan")

    return float(
        roc_auc_score(
            error_labels,
            np.asarray(uncertainty),
        )
    )


def save_risk_coverage_curve(
    results_df,
    output_path,
):
    ranked = results_df.sort_values(
        [
            "epistemic_uncertainty",
            "confidence",
        ],
        ascending=[True, False],
    ).reset_index(drop=True)

    rows = []
    n = len(ranked)

    for coverage in np.linspace(0.10, 1.0, 19):
        k = max(1, int(round(coverage * n)))
        selected = ranked.iloc[:k]
        risk = 1.0 - float(selected["correct"].mean())
        rows.append(
            {
                "coverage": k / n,
                "risk": risk,
                "selective_accuracy": 1.0 - risk,
            }
        )

    curve = pd.DataFrame(rows)
    curve.to_csv(
        RESULT_DIRECTORY / "risk_coverage.csv",
        index=False,
    )

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(
        curve["coverage"] * 100.0,
        curve["risk"] * 100.0,
        marker="o",
    )
    axis.set_xlabel("Coverage (%)")
    axis.set_ylabel("Selective risk (%)")
    axis.set_title("Risk–coverage curve")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    return curve


def save_confidence_uncertainty_plot(
    results_df,
    output_path,
):
    figure, axis = plt.subplots(figsize=(7, 5))

    correct = results_df["correct"].astype(bool)

    axis.scatter(
        results_df.loc[correct, "confidence"],
        results_df.loc[
            correct,
            "epistemic_uncertainty",
        ],
        s=12,
        alpha=0.45,
        label="Correct",
    )

    axis.scatter(
        results_df.loc[~correct, "confidence"],
        results_df.loc[
            ~correct,
            "epistemic_uncertainty",
        ],
        s=16,
        alpha=0.65,
        label="Incorrect",
    )

    axis.axvline(
        CONFIDENCE_THRESHOLD,
        linestyle="--",
        linewidth=1.0,
    )
    axis.axhline(
        UNCERTAINTY_THRESHOLD,
        linestyle="--",
        linewidth=1.0,
    )

    axis.set_xlabel("Ensemble confidence")
    axis.set_ylabel("Epistemic uncertainty")
    axis.set_title("Confidence–uncertainty diagnostic")
    axis.legend()
    axis.grid(alpha=0.20)
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_reliability_diagram(
    reliability_df,
    output_path,
):
    valid = reliability_df.dropna(
        subset=["mean_confidence", "accuracy"]
    )

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.0,
        label="Perfect calibration",
    )
    axis.plot(
        valid["mean_confidence"],
        valid["accuracy"],
        marker="o",
        label="AstroGuard-QFV",
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Mean confidence")
    axis.set_ylabel("Empirical accuracy")
    axis.set_title("Reliability diagram")
    axis.legend()
    axis.grid(alpha=0.20)
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def print_manuscript_values(summary):
    ds = summary["dataset"]
    ens = summary["ensemble"]
    gate = summary["runtime_assurance"]

    print("\n" + "=" * 78)
    print("VALUES TO COPY INTO THE MANUSCRIPT")
    print("=" * 78)
    print(f"NTotal       = {ds['total']}")
    print(f"NTrain       = {ds['train']}")
    print(f"NVal         = {ds['validation']}")
    print(f"NTest        = {ds['test']}")
    print(f"Accuracy     = {100.0 * ens['accuracy']:.2f}%")
    print(f"MacroP       = {100.0 * ens['macro_precision']:.2f}%")
    print(f"MacroR       = {100.0 * ens['macro_recall']:.2f}%")
    print(f"MacroF       = {100.0 * ens['macro_f1']:.2f}%")

    if ens["macro_auroc"] is None:
        print("AUROC        = unavailable")
    else:
        print(f"AUROC        = {ens['macro_auroc']:.4f}")

    print(f"ECE          = {ens['ece']:.4f}")

    if ens["error_detection_auroc"] is None:
        print("Error AUROC  = unavailable")
    else:
        print(
            f"Error AUROC  = "
            f"{ens['error_detection_auroc']:.4f}"
        )

    print(f"NReleased    = {gate['released_objects']}")
    print(f"Coverage     = {100.0 * gate['coverage']:.2f}%")

    if gate["released_accuracy"] is None:
        print("ReleaseAcc   = unavailable")
    else:
        print(
            f"ReleaseAcc   = "
            f"{100.0 * gate['released_accuracy']:.2f}%"
        )

    print(
        f"UnsafeRate   = "
        f"{100.0 * gate['unsafe_release_rate']:.2f}%"
    )
    print("=" * 78)


# =============================================================================
# RELEASE GATE
# =============================================================================

def release_gate(
    confidence,
    uncertainty,
    quality,
    provenance_ok,
):
    if not provenance_ok:
        return "WITHHOLD_PROVENANCE"

    if quality < QUALITY_THRESHOLD:
        return "HUMAN_REVIEW_QUALITY"

    if uncertainty > UNCERTAINTY_THRESHOLD:
        return "HUMAN_REVIEW_UNCERTAINTY"

    if confidence < CONFIDENCE_THRESHOLD:
        return "HUMAN_REVIEW_CONFIDENCE"

    return "RELEASE"


# =============================================================================
# MAIN
# =============================================================================

def main():
    prepare_environment()
    device = torch.device("cpu")

    print("Device:", device)

    client = Alerce()

    processed_path = (
        CACHE_DIRECTORY / f"processed_records_{TARGET_PER_CLASS}_per_class.npy"
    )

    processed_records = None

    if processed_path.exists():
        print("Loading target-specific cached processed data.")
        candidate_records = np.load(
            processed_path,
            allow_pickle=True,
        ).tolist()

        cached_counts = pd.Series(
            [r["label"] for r in candidate_records]
        ).value_counts().to_dict()

        cache_is_complete = all(
            cached_counts.get(name, 0) >= TARGET_PER_CLASS
            for name in CLASS_GROUPS
        )

        if cache_is_complete:
            processed_records = candidate_records
            print("Cached counts:", cached_counts)
        else:
            print(
                "Cached dataset is incomplete for the current target; "
                "rebuilding it."
            )

    if processed_records is None:
        objects_df = build_candidate_table(client)
        processed_records = build_processed_records(
            client,
            objects_df,
        )

    if len(processed_records) < 100:
        raise RuntimeError(
            "The final dataset contains fewer than "
            "100 usable light curves."
        )

    (
        label_encoder,
        encoded_labels,
        train_indices,
        validation_indices,
        test_indices,
        train_loader,
        validation_loader,
        test_loader,
    ) = create_splits_and_loaders(
        processed_records
    )

    number_of_classes = len(
        label_encoder.classes_
    )

    print("Total:", len(processed_records))
    print("Train:", len(train_indices))
    print("Validation:", len(validation_indices))
    print("Test:", len(test_indices))
    print(
        "Classes:",
        label_encoder.classes_.tolist(),
    )

    classification_loss_function = (
        make_classification_loss(
            encoded_labels,
            train_indices,
            number_of_classes,
            device,
        )
    )

    models = []
    validation_f1_scores = []

    for model_number, model_seed in enumerate(
        ENSEMBLE_SEEDS,
        start=1,
    ):
        print("\n" + "=" * 70)
        print(
            f"Training model "
            f"{model_number}/{len(ENSEMBLE_SEEDS)}"
        )

        random.seed(model_seed)
        np.random.seed(model_seed)
        torch.manual_seed(model_seed)

        model = HybridQuantumTransformer(
            number_of_classes
        ).to(device)

        model, history_df = train_model(
            model,
            train_loader,
            validation_loader,
            classification_loss_function,
            device,
        )

        validation = evaluate(
            model,
            validation_loader,
            classification_loss_function,
            device,
        )

        validation_f1_scores.append(
            validation["macro_f1"]
        )

        models.append(model)

        torch.save(
            model.state_dict(),
            MODEL_DIRECTORY
            / f"ensemble_model_{model_number}.pt",
        )

        history_df.to_csv(
            RESULT_DIRECTORY
            / f"training_history_{model_number}.csv",
            index=False,
        )

    test_outputs = [
        predict_model(
            model,
            test_loader,
            device,
        )
        for model in models
    ]

    probability_tensor = np.stack(
        [
            result["probabilities"]
            for result in test_outputs
        ],
        axis=0,
    )

    ensemble_weights = np.asarray(
        validation_f1_scores,
        dtype=np.float64,
    )

    if ensemble_weights.sum() <= 0:
        ensemble_weights = np.ones_like(
            ensemble_weights
        )

    ensemble_weights /= ensemble_weights.sum()

    mean_probabilities = np.average(
        probability_tensor,
        axis=0,
        weights=ensemble_weights,
    )

    predictions = mean_probabilities.argmax(
        axis=1
    )
    confidence = mean_probabilities.max(
        axis=1
    )

    true_labels = test_outputs[0]["labels"]
    object_ids = test_outputs[0]["oids"]
    quality_scores = test_outputs[0]["quality"]

    epistemic_uncertainty = (
        probability_tensor.var(axis=0)
        .mean(axis=1)
    )

    predictive_entropy = -np.sum(
        mean_probabilities
        * np.log(mean_probabilities + 1e-12),
        axis=1,
    )

    ece, reliability_df = expected_calibration_error(
        true_labels,
        predictions,
        confidence,
        n_bins=15,
    )

    error_detection_auroc = uncertainty_error_auroc(
        true_labels,
        predictions,
        epistemic_uncertainty,
    )

    reliability_df.to_csv(
        RESULT_DIRECTORY / "reliability_bins.csv",
        index=False,
    )

    accuracy = accuracy_score(
        true_labels,
        predictions,
    )
    precision = precision_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    recall = recall_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    macro_f1 = f1_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    try:
        auroc = roc_auc_score(
            true_labels,
            mean_probabilities,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        auroc = float("nan")

    results_df = pd.DataFrame(
        {
            "oid": object_ids,
            "true_label": (
                label_encoder.inverse_transform(
                    true_labels
                )
            ),
            "predicted_label": (
                label_encoder.inverse_transform(
                    predictions
                )
            ),
            "confidence": confidence,
            "epistemic_uncertainty": (
                epistemic_uncertainty
            ),
            "predictive_entropy": (
                predictive_entropy
            ),
            "quality_score": quality_scores,
            "correct": (
                true_labels == predictions
            ),
        }
    )

    for class_index, class_name in enumerate(
        label_encoder.classes_
    ):
        safe_name = (
            str(class_name)
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )
        results_df[
            f"prob_{safe_name}"
        ] = mean_probabilities[:, class_index]

    results_df["provenance_ok"] = (
        results_df["oid"]
        .astype(str)
        .str.startswith("ZTF")
    )

    results_df["decision"] = (
        results_df.apply(
            lambda row: release_gate(
                row["confidence"],
                row["epistemic_uncertainty"],
                row["quality_score"],
                row["provenance_ok"],
            ),
            axis=1,
        )
    )

    released = results_df[
        results_df["decision"] == "RELEASE"
    ]

    released_accuracy = (
        float(released["correct"].mean())
        if len(released) > 0
        else None
    )

    coverage = (
        float(len(released) / len(results_df))
        if len(results_df) > 0
        else 0.0
    )

    unsafe_release_mask = (
        (results_df["decision"] == "RELEASE")
        & (
            (~results_df["provenance_ok"])
            | (
                results_df["quality_score"]
                < QUALITY_THRESHOLD
            )
            | (
                results_df["epistemic_uncertainty"]
                > UNCERTAINTY_THRESHOLD
            )
            | (
                results_df["confidence"]
                < CONFIDENCE_THRESHOLD
            )
        )
    )

    unsafe_release_rate = (
        float(unsafe_release_mask.mean())
        if len(results_df) > 0
        else 0.0
    )

    results_df.to_csv(
        RESULT_DIRECTORY
        / "ensemble_predictions.csv",
        index=False,
    )

    confusion = confusion_matrix(
        true_labels,
        predictions,
    )

    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    ConfusionMatrixDisplay(
        confusion_matrix=confusion,
        display_labels=label_encoder.classes_,
    ).plot(
        ax=axis,
        colorbar=False,
    )

    plt.title(
        "Transformer–quantum ensemble"
    )
    plt.tight_layout()
    plt.savefig(
        PLOT_DIRECTORY
        / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    save_risk_coverage_curve(
        results_df,
        PLOT_DIRECTORY / "risk_coverage.png",
    )

    save_confidence_uncertainty_plot(
        results_df,
        PLOT_DIRECTORY
        / "confidence_uncertainty.png",
    )

    save_reliability_diagram(
        reliability_df,
        PLOT_DIRECTORY
        / "reliability_diagram.png",
    )

    summary = {
        "dataset": {
            "total": len(processed_records),
            "train": len(train_indices),
            "validation": len(
                validation_indices
            ),
            "test": len(test_indices),
            "classes": (
                label_encoder.classes_.tolist()
            ),
        },
        "ensemble": {
            "members": len(models),
            "weights": [
                float(value)
                for value in ensemble_weights
            ],
            "accuracy": float(accuracy),
            "macro_precision": float(
                precision
            ),
            "macro_recall": float(recall),
            "macro_f1": float(macro_f1),
            "macro_auroc": (
                None
                if np.isnan(auroc)
                else float(auroc)
            ),
            "mean_epistemic_uncertainty": (
                float(
                    epistemic_uncertainty.mean()
                )
            ),
            "mean_predictive_entropy": (
                float(
                    predictive_entropy.mean()
                )
            ),
            "ece": float(ece),
            "error_detection_auroc": (
                None
                if np.isnan(error_detection_auroc)
                else float(error_detection_auroc)
            ),
        },
        "runtime_assurance": {
            "released_objects": int(
                len(released)
            ),
            "released_accuracy": (
                released_accuracy
            ),
            "coverage": float(coverage),
            "unsafe_release_rate": float(
                unsafe_release_rate
            ),
        },
    }

    with open(
        RESULT_DIRECTORY / "final_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(
        f"Macro precision: "
        f"{precision * 100:.2f}%"
    )
    print(
        f"Macro recall: "
        f"{recall * 100:.2f}%"
    )
    print(
        f"Macro F1: "
        f"{macro_f1 * 100:.2f}%"
    )

    if np.isnan(auroc):
        print("Macro AUROC: unavailable")
    else:
        print(f"Macro AUROC: {auroc:.4f}")

    print("Released objects:", len(released))

    if released_accuracy is None:
        print("Released accuracy: unavailable")
    else:
        print(
            f"Released accuracy: "
            f"{released_accuracy * 100:.2f}%"
        )

    print(
        classification_report(
            true_labels,
            predictions,
            target_names=(
                label_encoder.classes_
            ),
            zero_division=0,
        )
    )

    print("Outputs:", OUTPUT_DIRECTORY)

    print_manuscript_values(summary)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Execution interrupted.")
        sys.exit(130)
    except Exception as error:
        print("Fatal error:", error)
        raise
