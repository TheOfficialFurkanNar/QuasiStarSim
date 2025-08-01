# tests/test_constants.py

import re
import pytest
import constants

def test_version_format():
    """
    VERSION sabiti SemVer (X.Y.Z) formatında olmalı.
    """
    assert hasattr(constants, "VERSION"), "VERSION tanımlı değil"
    semver_pattern = r"^\d+\.\d+\.\d+$"
    assert re.match(semver_pattern, constants.VERSION), \
           f"VERSION ‘{constants.VERSION}’ SemVer kalıbına uymuyor"

def test_default_log_level():
    """
    DEFAULT_LOG_LEVEL sabiti geçerli bir log seviyesi içermeli.
    """
    assert hasattr(constants, "DEFAULT_LOG_LEVEL"), "DEFAULT_LOG_LEVEL tanımlı değil"
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    assert constants.DEFAULT_LOG_LEVEL in valid_levels, \
           f"Beklenmeyen log seviyesi: {constants.DEFAULT_LOG_LEVEL}"

def test_supported_modules_list():
    """
    SUPPORTED_MODULES sabiti en az bir modül ismi içeren liste olmalı.
    """
    assert hasattr(constants, "SUPPORTED_MODULES"), "SUPPORTED_MODULES tanımlı değil"
    assert isinstance(constants.SUPPORTED_MODULES, list), \
           "SUPPORTED_MODULES bir liste değil"
    assert constants.SUPPORTED_MODULES, "SUPPORTED_MODULES listesi boş"