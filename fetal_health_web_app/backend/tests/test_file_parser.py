import io
import pytest
import numpy as np
import pandas as pd
from core.file_parser import parse_csv, extract_medical_metadata, validate_duration

# 4 rows, 1 missing FHR (=0), t_sec spans 0.0-0.75
SAMPLE_CSV = b"t_sec,FHR,UC\n0.0,150.5,7.0\n0.25,151.0,8.5\n0.5,0.0,9.0\n0.75,152.0,7.5\n"


def test_parse_csv_returns_dataframe():
    df = parse_csv(SAMPLE_CSV)
    assert list(df.columns) >= ["FHR", "UC"]
    assert len(df) == 4


def test_parse_csv_missing_uc_raises():
    bad = b"t_sec,FHR\n0.0,150.0\n"
    with pytest.raises(ValueError, match="UC"):
        parse_csv(bad)


def test_parse_csv_missing_fhr_raises():
    bad = b"t_sec,UC\n0.0,5.0\n"
    with pytest.raises(ValueError, match="FHR"):
        parse_csv(bad)


def test_extract_missing_signal_pct():
    df = parse_csv(SAMPLE_CSV)
    medical = extract_medical_metadata(df)
    assert medical.missing_signal_pct == 25.0  # 1 of 4 rows has FHR=0


def test_extract_fhr_mean():
    df = parse_csv(SAMPLE_CSV)
    medical = extract_medical_metadata(df)
    # valid FHR rows: 150.5, 151.0, 152.0 -> mean = 151.1667
    assert medical.fhr_mean is not None
    assert abs(medical.fhr_mean - 151.17) < 0.1


def test_extract_uc_available():
    df = parse_csv(SAMPLE_CSV)
    medical = extract_medical_metadata(df)
    assert medical.uc_available is True


def test_extract_duration_minutes():
    df = parse_csv(SAMPLE_CSV)
    medical = extract_medical_metadata(df)
    # t_sec goes from 0.0 to 0.75 -> duration = 0.75 sec = 0.0125 min
    assert medical.recording_duration_min is not None
    assert abs(medical.recording_duration_min - round(0.75 / 60, 2)) < 0.001


def test_validate_duration_ok():
    df = parse_csv(SAMPLE_CSV)
    medical = extract_medical_metadata(df)
    validate_duration(medical)  # should not raise


def test_validate_duration_too_long():
    n = int(91 * 60 * 4)  # 91 minutes at 4 Hz
    t = np.arange(n, dtype=np.float32) * 0.25
    fhr = np.full(n, 150.0, dtype=np.float32)
    uc = np.full(n, 5.0, dtype=np.float32)
    csv_bytes = pd.DataFrame({"t_sec": t, "FHR": fhr, "UC": uc}).to_csv(index=False).encode()
    df = parse_csv(csv_bytes)
    medical = extract_medical_metadata(df)
    with pytest.raises(ValueError, match="too long"):
        validate_duration(medical)



def _to_csv_bytes(df):
    return df.to_csv(index=False).encode()


def test_all_zero_fhr_is_100_percent_missing_and_mean_is_none():
    csv_bytes = _to_csv_bytes(pd.DataFrame({
        "t_sec": [0.0, 0.25, 0.5, 0.75],
        "FHR": [0.0, 0.0, 0.0, 0.0],
        "UC": [5.0, 5.0, 5.0, 5.0],
    }))

    df = parse_csv(csv_bytes)
    medical = extract_medical_metadata(df)

    assert medical.missing_signal_pct == 100.0
    assert medical.fhr_mean is None
    assert medical.fhr_std is None


def test_nan_fhr_is_counted_as_missing_signal():
    csv_bytes = _to_csv_bytes(pd.DataFrame({
        "t_sec": [0.0, 0.25, 0.5, 0.75],
        "FHR": [150.0, np.nan, 151.0, 0.0],
        "UC": [5.0, 5.0, 5.0, 5.0],
    }))

    df = parse_csv(csv_bytes)
    medical = extract_medical_metadata(df)

    assert medical.missing_signal_pct == 50.0
    assert medical.fhr_mean is not None
    assert abs(medical.fhr_mean - 150.5) < 0.01


def test_negative_fhr_is_counted_as_missing_signal():
    csv_bytes = _to_csv_bytes(pd.DataFrame({
        "t_sec": [0.0, 0.25, 0.5, 0.75],
        "FHR": [150.0, -1.0, 151.0, 0.0],
        "UC": [5.0, 5.0, 5.0, 5.0],
    }))

    df = parse_csv(csv_bytes)
    medical = extract_medical_metadata(df)

    assert medical.missing_signal_pct == 50.0
    assert medical.fhr_mean is not None
    assert abs(medical.fhr_mean - 150.5) < 0.01


def test_uc_available_false_when_all_uc_missing_or_zero():
    csv_bytes = _to_csv_bytes(pd.DataFrame({
        "t_sec": [0.0, 0.25, 0.5],
        "FHR": [140.0, 141.0, 142.0],
        "UC": [0.0, 0.0, 0.0],
    }))

    df = parse_csv(csv_bytes)
    medical = extract_medical_metadata(df)

    assert medical.uc_available is False


def test_duration_exactly_90_minutes_is_allowed():
    csv_bytes = _to_csv_bytes(pd.DataFrame({
        "t_sec": [0.0, 5400.0],
        "FHR": [140.0, 141.0],
        "UC": [5.0, 5.0],
    }))

    df = parse_csv(csv_bytes)
    medical = extract_medical_metadata(df)

    assert medical.recording_duration_min == 90.0
    validate_duration(medical)


def test_duration_just_over_90_minutes_is_rejected():
    csv_bytes = _to_csv_bytes(pd.DataFrame({
        "t_sec": [0.0, 5401.0],  # 90.017 min — unambiguously over 90 after rounding
        "FHR": [140.0, 141.0],
        "UC": [5.0, 5.0],
    }))

    df = parse_csv(csv_bytes)
    medical = extract_medical_metadata(df)

    with pytest.raises(ValueError, match="too long"):
        validate_duration(medical)


def test_parse_csv_empty_file_raises_value_error():
    bad = b""

    with pytest.raises(ValueError):
        parse_csv(bad)