"""Typed provider/model catalog shared by API presentation and routing policy."""

from __future__ import annotations

import os
from dataclasses import dataclass

from semantic_text2sql.models import GROQ_QWEN_MODEL, SOTA_GPT_MODEL, ModelOption, ModelProvider


@dataclass(frozen=True)
class ProviderModelSpec:
    provider: ModelProvider
    model: str
    api_key_env: str
    enabled_env: str | None = None
    supports_sql: bool = True
    supports_reasoning: bool = True

    def option(self) -> ModelOption:
        reason: str | None = None
        if not os.environ.get(self.api_key_env):
            reason = f"{self.api_key_env} is not set in the project environment."
        elif self.enabled_env and os.environ.get(self.enabled_env, "false").casefold() != "true":
            reason = (
                f"{self.provider} is disabled because access has not been verified; "
                f"set {self.enabled_env}=true after verification."
            )
        return ModelOption(
            provider=self.provider,
            model=self.model,
            local=False,
            configured=reason is None,
            unavailable_reason=reason,
        )


MODEL_CATALOG = (
    ProviderModelSpec("sota", SOTA_GPT_MODEL, "SOTA_API_KEY", "SOTA_ENABLED"),
    ProviderModelSpec("justdowork", "gpt-5.6-sol", "JUSTDOWORK_API_KEY", "JUSTDOWORK_ENABLED"),
    ProviderModelSpec("justdowork", "gpt-5.6-luna", "JUSTDOWORK_API_KEY", "JUSTDOWORK_ENABLED"),
    ProviderModelSpec("justdowork", "gpt-5.6-terra", "JUSTDOWORK_API_KEY", "JUSTDOWORK_ENABLED"),
    ProviderModelSpec("groq", GROQ_QWEN_MODEL, "GROQ_API_KEY"),
)


def configured_model_options() -> list[ModelOption]:
    return [spec.option() for spec in MODEL_CATALOG]
