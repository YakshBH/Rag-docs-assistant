from src.ingest.parse_pdf import _repair_mojibake


def test_extraction_normalizes_ligatures():
    assert _repair_mojibake("con\ufb01dently") == "confidently"
    assert _repair_mojibake("con" + bytes([0xEF, 0xAC]).decode("latin1") + "dently") == "confidently"


def test_extraction_repairs_common_mojibake():
    expected = chr(0x201C) + "hallucinations" + chr(0x201D)
    mojibake = bytes([0xE2, 0x80, 0x9C]).decode("latin1")
    mojibake += "hallucinations"
    mojibake += bytes([0xE2, 0x80, 0x9D]).decode("latin1")
    assert _repair_mojibake(mojibake) == expected
