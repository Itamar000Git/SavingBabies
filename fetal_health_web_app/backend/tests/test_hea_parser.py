import os
import tempfile
import pytest
from core.hea_parser import parse_hea_file, parse_outcome_fields

SAMPLE_HEA = """\
1001 2 4 19200
1001.dat 16 100(0)/bpm 12 0 15050 20101 0 FHR
1001.dat 16 100/nd 12 0 700 378 0 UC

#----- Additional parameters for record 1001

#-- Outcome measures
#pH           7.14
#BDecf        8.14

#-- Fetus/Neonate descriptors
#Gest. weeks  37
#Weight(g)    2660
#Sex          2
#Apgar1       6
#Apgar5       8

#-- Maternal (risk-)factors
#Age          32
#Gravidity    1
#Parity       0
#Diabetes     1
#Hypertension 0
#Preeclampsia 0
"""


@pytest.fixture
def hea_file(tmp_path):
    p = tmp_path / "1001.hea"
    p.write_text(SAMPLE_HEA)
    return str(p)


def test_parse_baby_gestational_weeks(hea_file):
    baby, _ = parse_hea_file(hea_file, "1001")
    assert baby.gestational_weeks == 37


def test_parse_baby_weight(hea_file):
    baby, _ = parse_hea_file(hea_file, "1001")
    assert baby.weight_g == 2660


def test_parse_baby_sex_female(hea_file):
    baby, _ = parse_hea_file(hea_file, "1001")
    assert baby.sex == "Female"


def test_parse_baby_apgar(hea_file):
    baby, _ = parse_hea_file(hea_file, "1001")
    assert baby.apgar1 == 6
    assert baby.apgar5 == 8


def test_parse_baby_id(hea_file):
    baby, _ = parse_hea_file(hea_file, "1001")
    assert baby.baby_id == "1001"


def test_parse_mother_age(hea_file):
    _, mother = parse_hea_file(hea_file, "1001")
    assert mother.mother_age == 32


def test_parse_mother_gravidity(hea_file):
    _, mother = parse_hea_file(hea_file, "1001")
    assert mother.gravidity == 1
    assert mother.parity == 0


def test_parse_mother_diabetes_true(hea_file):
    _, mother = parse_hea_file(hea_file, "1001")
    assert mother.diabetes is True


def test_parse_mother_hypertension_false(hea_file):
    _, mother = parse_hea_file(hea_file, "1001")
    assert mother.hypertension is False


def test_missing_file_returns_defaults():
    baby, mother = parse_hea_file("/nonexistent/path.hea", "9999")
    assert baby.baby_id == "9999"
    assert baby.gestational_weeks is None
    assert mother.mother_age is None

def test_parse_hea_missing_fields_return_none(tmp_path):
    p = tmp_path / "2001.hea"
    p.write_text("""\
2001 2 4 19200
2001.dat 16 100(0)/bpm 12 0 15050 20101 0 FHR
2001.dat 16 100/nd 12 0 700 378 0 UC
""")

    baby, mother = parse_hea_file(str(p), "2001")

    assert baby.baby_id == "2001"
    assert baby.gestational_weeks is None
    assert baby.weight_g is None
    assert mother.mother_age is None
    assert mother.diabetes is None


def test_parse_hea_sex_male(tmp_path):
    p = tmp_path / "2002.hea"
    p.write_text("""\
#Gest. weeks  38
#Sex          1
""")

    baby, _ = parse_hea_file(str(p), "2002")

    assert baby.sex == "Male"


def test_parse_hea_unknown_sex_returns_unknown_or_none(tmp_path):
    p = tmp_path / "2003.hea"
    p.write_text("""\
#Gest. weeks  38
#Sex          9
""")

    baby, _ = parse_hea_file(str(p), "2003")

    assert baby.sex in ("Unknown", None)


def test_parse_hea_boolean_values_are_parsed_correctly(tmp_path):
    p = tmp_path / "2004.hea"
    p.write_text("""\
#Diabetes     0
#Hypertension 1
#Preeclampsia 0
""")

    _, mother = parse_hea_file(str(p), "2004")

    assert mother.diabetes is False
    assert mother.hypertension is True
    assert mother.preeclampsia is False


def test_parse_bdecf_present(hea_file):
    outcomes = parse_outcome_fields(hea_file)
    assert outcomes["bdecf"] == pytest.approx(8.14)


def test_parse_bdecf_missing_returns_none(tmp_path):
    p = tmp_path / "3001.hea"
    p.write_text("#pH           7.10\n#Gest. weeks  38\n")
    outcomes = parse_outcome_fields(str(p))
    assert outcomes["bdecf"] is None


def test_parse_bdecf_nonexistent_file_returns_none():
    outcomes = parse_outcome_fields("/nonexistent/path.hea")
    assert outcomes["bdecf"] is None


def test_parse_bdecf_is_float(hea_file):
    outcomes = parse_outcome_fields(hea_file)
    assert isinstance(outcomes["bdecf"], float)