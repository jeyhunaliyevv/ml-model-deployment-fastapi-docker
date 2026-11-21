from typing import Literal
from pydantic import BaseModel, Field, ConfigDict

class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    age: int = Field(ge=0, le=120, description="Age in years")
    sex: Literal["female", "male"]
    bmi: float = Field(gt=0, le=100, allow_inf_nan=False)
    children: int = Field(ge=0, le=20)
    smoker: Literal["yes", "no"]
    region: Literal["northeast", "northwest", "southeast", "southwest"]

class Prediction(BaseModel):
    high_cost_probability: float = Field(ge=0, le=1)
    predicted_high_cost: bool
    threshold_usd: float = 20000.0
    decision_threshold: float = 0.5
    disclaimer: str = "Educational prediction only; not a medical or underwriting decision."

class Health(BaseModel):
    status: str
    model_loaded: bool
