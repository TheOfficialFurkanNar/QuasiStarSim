# config_logging.py

import os
import sys
import logging
import logging.config
from logging.handlers import RotatingFileHandler
import yaml

# ---------------------------------------------------------------------------
# 1. YAML Konfigürasyonunu Yükleme
# ---------------------------------------------------------------------------
def _load_yaml_config(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------------------------
# 2. Klasör ve Dosya Ayarları
# ---------------------------------------------------------------------------
LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE       = os.getenv("LOG_FILE",       os.path.join(LOG_DIR, "app.log"))
ERROR_LOG_FILE = os.getenv("ERROR_LOG_FILE", os.path.join(LOG_DIR, "error.log"))
YAML_CONFIG    = os.getenv("LOG_CONFIG_YAML", "logging_config.yaml")

# ---------------------------------------------------------------------------
# 3. Default Logging Konfigürasyonu
# ---------------------------------------------------------------------------
default_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": os.getenv("LOG_LEVEL_CONSOLE", "INFO"),
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": os.getenv("LOG_LEVEL_FILE", "DEBUG"),
            "formatter": "standard",
            "filename": LOG_FILE,
            "mode": "a",
            "maxBytes": 10 * 1024 * 1024,  # 10 MB
            "backupCount": 5,
            "encoding": "utf-8",
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "standard",
            "filename": ERROR_LOG_FILE,
            "mode": "a",
            "maxBytes": 5 * 1024 * 1024,   # 5 MB
            "backupCount": 3,
            "encoding": "utf-8",
        },
    },
    "root": {
        "level": os.getenv("LOG_LEVEL_ROOT", "DEBUG"),
        "handlers": ["console", "file", "error_file"],
    },
}

# ---------------------------------------------------------------------------
# 4. Konfigürasyonu Yükle ve Uygula
# ---------------------------------------------------------------------------
if os.path.exists(YAML_CONFIG):
    try:
        cfg = _load_yaml_config(YAML_CONFIG)
        logging.config.dictConfig(cfg)
    except Exception as e:
        logging.config.dictConfig(default_config)
        logging.getLogger(__name__).error(
            "Failed to load YAML logging config, using default_config", exc_info=e
        )
else:
    logging.config.dictConfig(default_config)

# ---------------------------------------------------------------------------
# 5. Logger Alıcı
# ---------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger instance.

    Parameters
    ----------
    name : str
        Logger adı (genellikle __name__ kullanılır).

    Returns
    -------
    logging.Logger
        Konfigüre edilmiş logger nesnesi.
    """
    return logging.getLogger(name)

# ---------------------------------------------------------------------------
# 6. Manuel RotatingFileHandler Ekleme
# ---------------------------------------------------------------------------
# Burada, root logger’a ek bir RotatingFileHandler daha
# tanımlayıp INFO seviyesi için force etmek istiyoruz.

logger = get_logger(__name__)

# Formatter’ı dictConfig’te tanımlı formatla tutarlı olacak şekilde yeniden oluşturuyoruz.
formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

manual_handler = RotatingFileHandler(
    filename=os.path.join(LOG_DIR, "app.log"),
    maxBytes=5 * 1024 * 1024,   # 5 MB
    backupCount=5,
    encoding="utf-8",
)
manual_handler.setLevel(logging.INFO)
manual_handler.setFormatter(formatter)

logger.addHandler(manual_handler)