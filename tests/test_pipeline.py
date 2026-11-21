import pandas as pd
import pytest
from src.pipeline import engineer, make_target

VALID = {"age": 35, "sex": "female", "bmi": 27.1, "children": 1,
         "smoker": "no", "region": "northeast"}

def test_engineer_adds_features():
    x = engineer(pd.DataFrame([VALID]))
    assert list(x["obese"]) == [0]
    assert list(x["smoker_obese"]) == [0]

def test_target_threshold():
    assert make_target(pd.Series([19999, 20000, 20001])).tolist() == [0,1,1]

def test_missing_column():
    with pytest.raises(ValueError):
        engineer(pd.DataFrame([{"age":35}]))
