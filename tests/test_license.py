import pytest

from batterygemma.screen.license import ALLOWED, FLAGGED, REJECTED, UNKNOWN, evaluate_license, normalize_license

ALLOW = ["CC0", "CC-BY", "public-domain"]
FLAG = ["CC-BY-SA"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://creativecommons.org/licenses/by/4.0/", "CC-BY-4.0"),  # Crossref
        ("http://creativecommons.org/licenses/by-nc-nd/4.0/", "CC-BY-NC-ND-4.0"),
        ("http://creativecommons.org/publicdomain/zero/1.0/", "CC0-1.0"),  # arXiv OAI-PMH
        ("https://creativecommons.org/publicdomain/mark/1.0/", "public-domain"),
        ("cc-by", "CC-BY"),  # OpenAlex
        ("cc by-nc-nd", "CC-BY-NC-ND"),  # Europe PMC
        ("CC BY-SA 4.0", "CC-BY-SA-4.0"),
        ("cc0", "CC0-1.0"),
        ("public-domain", "public-domain"),
        ("http://arxiv.org/licenses/nonexclusive-distrib/1.0/", "arXiv-nonexclusive-1.0"),
        ("publisher-specific-oa", "other:publisher-specific-oa"),
        ("", None),
        (None, None),
    ],
)
def test_normalize_license(raw, expected):
    assert normalize_license(raw) == expected


@pytest.mark.parametrize(
    ("license_id", "decision"),
    [
        ("CC-BY-4.0", ALLOWED),
        ("CC-BY", ALLOWED),
        ("CC0-1.0", ALLOWED),
        ("public-domain", ALLOWED),
        ("CC-BY-SA-4.0", FLAGGED),
        ("CC-BY-NC-4.0", REJECTED),
        ("CC-BY-NC-ND", REJECTED),
        ("arXiv-nonexclusive-1.0", REJECTED),
        ("other:publisher-specific-oa", REJECTED),
        (None, UNKNOWN),
    ],
)
def test_evaluate_license(license_id, decision):
    assert evaluate_license(license_id, ALLOW, FLAG) == decision
