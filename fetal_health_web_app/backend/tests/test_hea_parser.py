import os
import tempfile
import pytest
from core.hea_parser import parse_hea_file

SAMPLE_HEA = """\
1001 2 4 19200
1001.dat 16 100(0)/bpm 12 0 15050 20101 0 FHR
1001.dat 16 100/nd 12 0 700 378 0 UC

#----- Additional parameters for record 1001

#-- Outcome measures
#pH           7.14

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
