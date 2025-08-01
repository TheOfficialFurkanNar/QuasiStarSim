# tests/test_config_logging.py

import logging
import io
import re
import pytest

import config_logging

@pytest.fixture(autouse=True)
def reset_logging():
    """
    Her test öncesi root logger'ı sıfırla
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.root.setLevel(logging.NOTSET)
    yield

def test_configure_sets_root_level():
    # DEBUG seviyesinde yapılandır
    config_logging.setup_logging(level="DEBUG")
    assert logging.root.level == logging.DEBUG

def test_logger_inherits_root_level():
    config_logging.setup_logging(level="WARNING")
    logger = config_logging.get_logger("module")
    assert logger.level == logging.WARNING

def test_default_stream_handler_exists():
    config_logging.setup_logging()
    logger = config_logging.get_logger("module")
    has_stream = any(isinstance(h, logging.StreamHandler)
                     for h in logger.handlers)
    assert has_stream

def test_log_output_format():
    # Özel StringIO handler ile çıktıyı yakala
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    fmt = "%(levelname)s:%(name)s:%(message)s"
    handler.setFormatter(logging.Formatter(fmt))

    config_logging.setup_logging(handlers=[handler])
    logger = config_logging.get_logger("mod")
    logger.error("Error occurred")

    output = stream.getvalue().strip()
    assert re.match(r"ERROR:mod:Error occurred", output)