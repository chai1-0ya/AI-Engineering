from config_utils.models import ModelConfig


class ConfigurationError(ValueError):
    """Raise when configuration values are invalid."""


def parse_model_config(
    provider: str, model_name: str, temperature: float = 0.0
) -> ModelConfig:
    """Parse model configuration values and return a ModelConfig object."""
    if not provider.strip():
        raise ConfigurationError("Provider cannot be empty.")

    if not model_name.strip():
        raise ConfigurationError("Model name cannot be empty.")

    if not 0.0 <= temperature <= 2.0:
        raise ConfigurationError("Temperature must be between 0.0 and 2.0.")

    return ModelConfig(
        provider=provider, model_name=model_name, temperature=temperature
    )
