import mlflow.sklearn
import pytest
import pandas as pd

@pytest.fixture
def load_model():
    model_path = "./model"
    model = mlflow.sklearn.load_model(model_path)
    return model

def test_model_loading(load_model):
    assert load_model is not None, "Failed to load the model!"

def test_model_inference(load_model):
    sample_input = pd.DataFrame({
        "MDC_PULS_OXIM_PULS_RATE_Result_min": [80, 85],
        "MDC_TEMP_Result_mean": [36.5, 36.7],
        "MDC_PULS_OXIM_PULS_RATE_Result_mean": [82.5, 83],
        "HR_to_RR_Ratio": [4.2, 4.5]
    })
    predictions = load_model.predict(sample_input)
    assert len(predictions) == len(sample_input), "Inference failed!"
