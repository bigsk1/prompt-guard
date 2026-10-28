"""Prompt injection scanning service."""

__version__ = "1.0.0"

CONTRACT_VERSION = "1.0"
SERVICE_NAME = "prompt-guard"
AGGREGATION = "max"

DEFAULT_MODEL_ID = "Horizon-Labs/prompt-injection-guard-base"
# Full commit SHA. Do not point this at a branch or tag.
DEFAULT_MODEL_REVISION = "f95005bbceac345aafec1620719857c6202eabd5"

EXPECTED_ID2LABEL = {"0": "SAFE", "1": "INJECTION"}
EXPECTED_LABEL2ID = {"SAFE": 0, "INJECTION": 1}
INJECTION_LABEL = "INJECTION"
INJECTION_INDEX = 1
