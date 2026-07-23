import pytest
import os

def test_models_directory_exists():
    assert os.path.exists("outputs/models")

def test_model_files_exist():
    model_files = [f for f in os.listdir("outputs/models") if f.endswith((".pkl", ".joblib", ".h5", ".pt", ".json"))]
    assert len(model_files) > 0
