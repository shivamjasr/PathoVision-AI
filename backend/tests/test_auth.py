from backend.app.auth.service import create_access_token, hash_password, verify_password, current_user_from_token


def test_password_hash_roundtrip():
    password = "correct-horse-battery"
    hashed = hash_password(password)
    assert hashed != password
    assert verify_password(password, hashed)
    assert not verify_password("wrong-password", hashed)


def test_jwt_roundtrip():
    token = create_access_token("alice", "user")
    payload = current_user_from_token(token)
    assert payload == {"username": "alice", "role": "user"}
