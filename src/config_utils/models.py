from dataclasses import dataclass


@dataclass
class ModelConfig:
    """store configuration for an LLM."""

    provider: str
    model_name: str
    temperature: float = 0.0
