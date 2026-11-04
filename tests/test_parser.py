import pytest

from config_utils.models import ModelConfig
from config_utils.parser import ConfigurationError, parse_model_config


def test_parse_valid_config() -> None:
    result = parse_model_config(
        "google_genai",
        "gemini-3.1-flash-lite",
        0.7,
    )

    assert result == ModelConfig(
        provider="google_genai",
        model_name="gemini-3.1-flash-lite",
        temperature=0.7,
    )


def test_default_temperature() -> None:
    result = parse_model_config(
        "google_genai",
        "gemini-3.1-flash-lite",
    )

    assert result.temperature == 0.0


def test_empty_provider_raises_error() -> None:
    with pytest.raises(
        ConfigurationError,
        match="Provider cannot be empty",
    ):
        parse_model_config("", "gemini-3.1-flash-lite")


def test_whitespace_provider_raises_error() -> None:
    with pytest.raises(
        ConfigurationError,
        match="Provider cannot be empty",
    ):
        parse_model_config("   ", "gemini-3.1-flash-lite")


def test_empty_model_name_raises_error() -> None:
    with pytest.raises(
        ConfigurationError,
        match="Model name cannot be empty",
    ):
        parse_model_config("google_genai", "")


def test_whitespace_model_name_raises_error() -> None:
    with pytest.raises(
        ConfigurationError,
        match="Model name cannot be empty",
    ):
        parse_model_config("google_genai", "   ")


def test_minimum_temperature_is_valid() -> None:
    result = parse_model_config(
        "google_genai",
        "gemini-3.1-flash-lite",
        0.0,
    )

    assert result.temperature == 0.0


def test_maximum_temperature_is_valid() -> None:
    result = parse_model_config(
        "google_genai",
        "gemini-3.1-flash-lite",
        2.0,
    )

    assert result.temperature == 2.0


def test_temperature_below_range_raises_error() -> None:
    with pytest.raises(
        ConfigurationError,
        match="Temperature must be between 0.0 and 2.0",
    ):
        parse_model_config(
            "google_genai",
            "gemini-3.1-flash-lite",
            -0.1,
        )


def test_temperature_above_range_raises_error() -> None:
    with pytest.raises(
        ConfigurationError,
        match="Temperature must be between 0.0 and 2.0",
    ):
        parse_model_config(
            "google_genai",
            "gemini-3.1-flash-lite",
            2.1,
        )
