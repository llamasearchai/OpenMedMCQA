from pydantic import BaseModel, Field
import os
from pathlib import Path
from dotenv import load_dotenv


load_dotenv(override=False)


class Settings(BaseModel):
    openai_api_key: str = Field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    openai_api_base: str = Field(default_factory=lambda: os.environ.get("MEDAGENT_API_BASE", ""))
    openai_model: str = Field(default_factory=lambda: os.environ.get("MEDAGENT_MODEL", "gpt-4o-mini"))
    embedding_model: str = Field(default_factory=lambda: os.environ.get("MEDAGENT_EMBEDDING_MODEL", "allenai/scibert_scivocab_uncased"))
    embedding_backend: str = Field(default_factory=lambda: os.environ.get("MEDAGENT_EMBEDDING_BACKEND", "hf"))  # hf | openai
    embedding_api_base: str = Field(default_factory=lambda: os.environ.get("MEDAGENT_EMBEDDING_API_BASE", ""))
    db_path: str = Field(default_factory=lambda: os.environ.get("MEDAGENT_DB", "medagent.db"))
    index_dir: str = Field(default_factory=lambda: os.environ.get("MEDAGENT_INDEX_DIR", ".index"))
    max_ctx: int = Field(default_factory=lambda: int(os.environ.get("MEDAGENT_MAX_CTX", "5")))
    top_k: int = Field(default_factory=lambda: int(os.environ.get("MEDAGENT_TOP_K", "20")))
    light_tests: bool = Field(default_factory=lambda: os.environ.get("LIGHT_TESTS", "0") == "1")
    use_assistants: bool = Field(default_factory=lambda: os.environ.get("MEDAGENT_USE_ASSISTANTS", "0") == "1")
    use_function_calling: bool = Field(default_factory=lambda: os.environ.get("MEDAGENT_USE_FUNCTION_CALLING", "0") == "1")
    temperature: float = Field(default_factory=lambda: float(os.environ.get("MEDAGENT_TEMPERATURE", "0.0")))

    def ensure_dirs(self) -> None:
        Path(self.index_dir).mkdir(parents=True, exist_ok=True)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)


settings = Settings()

