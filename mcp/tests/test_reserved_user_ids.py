"""create_user_token must refuse the reserved sentinel identities.

A token whose user_id is LOCAL_USER_ID is treated as a full local admin by
_caller_privileges, so minting one is a privilege escalation (review Chain 1).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from _auth import (  # noqa: E402
    ANONYMOUS_USER_ID, LOCAL_USER_ID, RESERVED_USER_IDS,
    create_user_token, list_users,
)


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MOLAGENT_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("MOLAGENT_TOKEN_STORE_PATH", str(tmp_path / "auth_tokens.json"))


@pytest.mark.parametrize("reserved", sorted(RESERVED_USER_IDS))
def test_create_user_token_rejects_reserved(reserved):
    with pytest.raises(ValueError, match="reserved"):
        create_user_token(reserved)


def test_reserved_set_covers_both_sentinels():
    assert RESERVED_USER_IDS == frozenset({LOCAL_USER_ID, ANONYMOUS_USER_ID})


def test_no_token_is_stored_for_a_rejected_user_id():
    with pytest.raises(ValueError):
        create_user_token(LOCAL_USER_ID)
    assert all(u.get("user_id") != LOCAL_USER_ID for u in list_users())


def test_normal_user_id_still_works():
    token = create_user_token("alice")
    assert token.startswith("molagent_usr_")
    assert any(u.get("user_id") == "alice" for u in list_users())
