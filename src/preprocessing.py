"""Training-data preprocessing pipeline."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from .eda import ID_COLUMN, TARGET_COLUMN


@dataclass
class PreprocessingResult:
    """Processed train/test data and fitted preprocessing objects."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    target_encoder: LabelEncoder
    numeric_features: list[str]
    categorical_features: list[str]


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build the imputation, scaling, and one-hot encoding pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def preprocess_dataset(
    dataframe: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    id_column: str = ID_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
) -> PreprocessingResult:
    """Split and transform a labeled dataset using a fitted train-only pipeline."""
    X = dataframe.drop(columns=[target_column, id_column], errors="ignore")
    y_text = dataframe[target_column].astype(str)
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    X_train, X_test, y_train_text, y_test_text = train_test_split(
        X,
        y_text,
        test_size=test_size,
        random_state=random_state,
        stratify=y_text,
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_array = preprocessor.fit_transform(X_train)
    X_test_array = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    X_train_processed = pd.DataFrame(
        X_train_array.toarray() if hasattr(X_train_array, "toarray") else X_train_array,
        columns=feature_names,
        index=X_train.index,
    )
    X_test_processed = pd.DataFrame(
        X_test_array.toarray() if hasattr(X_test_array, "toarray") else X_test_array,
        columns=feature_names,
        index=X_test.index,
    )

    target_encoder = LabelEncoder()
    y_train = pd.Series(
        target_encoder.fit_transform(y_train_text),
        index=y_train_text.index,
        name=target_column,
    )
    y_test = pd.Series(
        target_encoder.transform(y_test_text),
        index=y_test_text.index,
        name=target_column,
    )
    return PreprocessingResult(
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor,
        target_encoder,
        numeric_features,
        categorical_features,
    )


def save_preprocessing_result(
    result: PreprocessingResult,
    output_dir: str | Path = "data/processed",
) -> Path:
    """Save processed data and fitted preprocessing artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result.X_train.to_csv(output_path / "X_train_processed.csv", index=False)
    result.X_test.to_csv(output_path / "X_test_processed.csv", index=False)
    result.y_train.to_csv(output_path / "y_train.csv", index=False)
    result.y_test.to_csv(output_path / "y_test.csv", index=False)
    joblib.dump(result.preprocessor, output_path / "preprocessor.joblib")
    joblib.dump(result.target_encoder, output_path / "target_label_encoder.joblib")
    return output_path
