from src.extract import _normalize


def test_normalize_trims_and_collapses():
    s = "Hello   world\n\nThis\tis  a  test."
    out = _normalize(s, max_chars=100)
    assert out == "Hello world This is a test."
    assert len(out) <= 100
