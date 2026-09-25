import pytest
from fastapi import HTTPException

from backend.app.security_utils import safe_filename


def test_safe_filename_strips_paths_and_special_chars():
    assert safe_filename("../../My slide 01.svs") == "My_slide_01.svs"


def test_safe_filename_rejects_unknown_extension():
    with pytest.raises(HTTPException) as exc:
        safe_filename("secret.txt")
    assert exc.value.status_code == 400
