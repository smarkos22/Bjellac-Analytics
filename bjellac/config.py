from pathlib import Path
import os

from dotenv import load_dotenv
from yaml import safe_load


def find_project_root() -> Path:
    current = Path(__file__).resolve().parent.parent
    if (current / "README.md").exists():
        return current
    raise FileNotFoundError("Cannot locate Bjellac project root (no README.md found)")


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
STAGING_DIR = DATA_DIR / "staging"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
UNSTRUCTURED_DIR = DATA_DIR / "unstructured"
DB_PATH = DATA_DIR / "bjellac.duckdb"
CONFIG_DIR = PROJECT_ROOT / "config"
SOURCES_YAML = CONFIG_DIR / "sources.yaml"

load_dotenv(PROJECT_ROOT / ".env")


def get_api_key(env_var: str) -> str:
    key = os.environ.get(env_var)
    if not key:
        raise EnvironmentError(
            f"{env_var} not set. Copy .env.example to .env and fill in your key."
        )
    return key


def load_sources_config() -> dict:
    with open(SOURCES_YAML) as f:
        return safe_load(f)


def ensure_dirs():
    for d in [
        RAW_DIR,
        STAGING_DIR,
        PROCESSED_DIR,
        FEATURES_DIR,
        UNSTRUCTURED_DIR / "inbox",
        UNSTRUCTURED_DIR / "processed",
        UNSTRUCTURED_DIR / "media",
        DATA_DIR / "exports",
    ]:
        d.mkdir(parents=True, exist_ok=True)
