import os
import pytest
import pandas as pd
from src.healthcare_pipeline import (
    find_dataset_file,
    load_and_preprocess,
    run_healthcare_pipeline
)

DATA_PATH = "data/healthcare_appointments_dataset.csv"

def test_find_dataset_file():
    resolved = find_dataset_file(DATA_PATH)
    assert os.path.exists(resolved), f"Arquivo não encontrado: {resolved}"

def test_load_and_preprocess():
    df = load_and_preprocess(DATA_PATH)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1200
    assert "attendance_status" in df.columns
    assert "specialty" in df.columns
    assert "lead_time_days" in df.columns
    assert "confirmed_digital" in df.columns

def test_run_healthcare_pipeline():
    model = run_healthcare_pipeline(DATA_PATH)
    assert model is not None
    assert hasattr(model, "predict_proba")
    assert hasattr(model, "feature_importances_")
