# This file is part of ipumspy.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/ipumspy

import gzip
import tempfile
from functools import partial
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from ipumspy import readers
from ipumspy.api.extract import MicrodataExtract


def _values(series: pd.Series) -> list:
    """Series values as a plain list, with any missing value as None"""
    return [None if pd.isna(v) else v for v in series.tolist()]


def _assert_dtypes_match_ddi(data: pd.DataFrame, ddi, overrides: Dict = None):
    """Every column should have the pandas type declared for it in the DDI"""
    overrides = overrides or {}
    expected = {
        col: overrides.get(col, ddi.get_variable_info(col).pandas_type)
        for col in data.columns
    }
    assert dict(data.dtypes) == expected


def _assert_cps_000006(data: pd.DataFrame):
    """Run all the checks for the data frame returned by our readers for rectangular files"""
    assert len(data) == 7668
    assert len(data.columns) == 8
    assert (data["YEAR"].iloc[:5] == 1962).all()
    assert (
        data["HWTSUPP"].iloc[:5]
        == np.array([1475.59, 1475.59, 1475.59, 1597.61, 1706.65])
    ).all()
    assert (
        data.dtypes.values
        == np.array(
            [
                pd.Int64Dtype(),
                pd.Int64Dtype(),
                float,
                pd.Int64Dtype(),
                pd.Int64Dtype(),
                pd.Int64Dtype(),
                float,
                pd.Int64Dtype(),
            ]
        )
    ).all()


def _assert_cps_00421_df(data: pd.DataFrame, ddi):
    """Run all the checks for the data frame returned by our readers for hierarchical files"""
    assert len(data) == 339278
    assert len(data.columns) == 14
    assert (data["YEAR"].iloc[:5] == 2022).all()
    assert _values(data["HWTFINL"].iloc[:5]) == [0.0, 1662.5757, None, None, None]
    assert _values(data["RECTYPE"].iloc[:5]) == ["H", "H", "P", "P", "P"]
    assert _values(data["PERNUM"].iloc[:5]) == [None, None, 1, 2, 3]
    _assert_dtypes_match_ddi(data, ddi)


def _assert_cps_00421_dict(data: Dict, ddi):
    """Run all the checks for the data frame returned by our readers for hierarchical files
    when a dictionary of data frames is requested"""
    assert set(data.keys()) == {"P", "H"}
    p_data = data["P"]
    h_data = data["H"]

    assert len(p_data) == 201993
    assert len(p_data.columns) == 9
    assert (p_data["YEAR"].iloc[:5] == 2022).all()
    assert _values(p_data["WTFINL"].iloc[:5]) == [
        1662.5757,
        1978.1985,
        1801.0842,
        1243.6042,
        2037.9611,
    ]
    assert (p_data["RECTYPE"].iloc[:5] == "P").all()
    assert _values(p_data["PERNUM"].iloc[:5]) == [1, 2, 3, 4, 1]
    _assert_dtypes_match_ddi(p_data, ddi)

    assert len(h_data) == 137285
    assert len(h_data.columns) == 8
    assert (h_data["YEAR"].iloc[:5] == 2022).all()
    assert _values(h_data["HWTFINL"].iloc[:5]) == [
        0.0,
        1662.5757,
        2037.9611,
        2094.5077,
        1970.825,
    ]
    assert (h_data["RECTYPE"].iloc[:5] == "H").all()
    assert _values(h_data["MISH"].iloc[:5]) == [7, 5, 1, 2, 1]
    _assert_dtypes_match_ddi(h_data, ddi)


def _assert_meps_000019(data: pd.DataFrame):
    """Run all the checks for the data frame returned by our readers for rectangular on R rectangular files"""
    assert len(data) == 103965
    assert len(data.columns) == 23
    assert (data["YEAR"].iloc[:5] == 2016).all()
    assert (
        data["SAQWEIGHT"].iloc[:5]
        == np.array(
            [14398.747070, 14398.747070, 14398.747070, 13439.433593, 13439.433593]
        )
    ).all()
    assert (
        data.dtypes.values
        == np.array(
            [
                pd.Int64Dtype(),
                pd.Int64Dtype(),
                pd.Int64Dtype(),
                "string[python]",
                "string[python]",
                pd.Int64Dtype(),
                pd.Int64Dtype(),
                pd.Int64Dtype(),
                pd.Int64Dtype(),
                pd.Int64Dtype(),
                pd.Int64Dtype(),
                pd.Int64Dtype(),
                float,
                float,
                float,
                pd.Int64Dtype(),
                pd.Int64Dtype(),
                pd.Int64Dtype(),
                "string[python]",
                pd.Int64Dtype(),
                "string[python]",
                pd.Int64Dtype(),
                pd.Int64Dtype(),
            ]
        )
    ).all()


def _assert_meps_00045_dict(data: Dict, ddi):
    """Run all the checks for the data frame returned by our readers for hierarchical MEPS files
    when a dictionary of data frames is requested"""
    assert set(data.keys()) == {"P", "R", "M", "F"}

    p_data = data["P"]
    assert len(p_data) == 34655
    assert len(p_data.columns) == 19
    assert (p_data["YEAR"].iloc[:5] == 2016).all()
    assert _values(p_data["SAQWEIGHT"].iloc[:5]) == [
        14398.74707,
        13439.433593,
        0.0,
        0.0,
        5559.980468,
    ]
    _assert_dtypes_match_ddi(p_data, ddi)

    r_data = data["R"]
    assert len(r_data) == 103965
    assert len(r_data.columns) == 11
    assert (r_data["YEARR"].iloc[:5] == 2016).all()
    _assert_dtypes_match_ddi(r_data, ddi)

    m_data = data["M"]
    assert len(m_data) == 137548
    assert len(m_data.columns) == 13
    assert (m_data["YEARM"].iloc[:5] == 2016).all()
    assert _values(m_data["NREFILLS"].iloc[:5]) == [1, 1, 2, 1, 1]
    _assert_dtypes_match_ddi(m_data, ddi)

    f_data = data["F"]
    assert len(f_data) == 319685
    assert len(f_data.columns) == 17
    assert (f_data["YEARF"].iloc[:5] == 2016).all()
    assert _values(f_data["RXDRGNAM"].iloc[:5]) == [
        "METRONIDAZOLE",
        "PROMETHAZINE",
        "RIFAXIMIN",
        "RIFAXIMIN",
        "HYDROCHLOROTHIAZIDE-LOSARTAN",
    ]
    _assert_dtypes_match_ddi(f_data, ddi)


def _assert_meps_00045_df(data: pd.DataFrame, ddi):
    """Run all the checks for the data frame returned by our readers for hierarchical files"""
    assert len(data) == 595853
    assert len(data.columns) == 51
    assert (data["SERIAL"].iloc[:5] == 1).all()
    assert _values(data["RECTYPE"].iloc[:6]) == ["P", "R", "R", "R", "M", "F"]
    # variables only exist on their own record type; other rows are missing
    assert _values(data["SAQWEIGHT"].iloc[:5]) == [14398.74707, None, None, None, None]
    assert _values(data["PREGNTRD"].iloc[:5]) == [None, 1, 1, 1, None]
    assert _values(data["MEPSIDM"].iloc[:6]) == [
        None,
        None,
        None,
        None,
        "2110001101",
        None,
    ]
    assert _values(data["RXDRGNAM"].iloc[:6]) == [
        None,
        None,
        None,
        None,
        None,
        "METRONIDAZOLE",
    ]
    _assert_dtypes_match_ddi(data, ddi)


def _assert_atus_00035_dict(data: Dict, ddi):
    """Run all the checks for the data frame returned by our readers for hierarchical files
    when a dictionary of data frames is requested"""
    assert set(data.keys()) == {"1", "2", "3"}
    h_data = data["1"]
    p_data = data["2"]
    a_data = data["3"]

    assert len(h_data) == 24336
    assert len(h_data.columns) == 5
    assert (h_data["YEAR"].iloc[:5] == 2016).all()
    assert _values(h_data["STATEFIP"].iloc[:5]) == [13, 51, 11, 26, 29]
    assert (h_data["RECTYPE"].iloc[:5] == 1).all()
    _assert_dtypes_match_ddi(h_data, ddi)

    assert len(p_data) == 24336
    assert len(p_data.columns) == 10
    assert (p_data["YEAR"].iloc[:5] == 2016).all()
    assert _values(p_data["WT06"].iloc[:5]) == [
        24588650.161504,
        5445941.065425,
        8782621.982064,
        3035909.94892,
        6978586.369092,
    ]
    assert (p_data["RECTYPE"].iloc[:5] == 2).all()
    # WT06 has decimals in the data, so the reader upcasts it from Int64
    _assert_dtypes_match_ddi(p_data, ddi, overrides={"WT06": pd.Float64Dtype()})

    assert len(a_data) == 207213
    assert len(a_data.columns) == 8
    assert (a_data["YEAR"].iloc[:5] == 2016).all()
    assert _values(a_data["ACTIVITY"].iloc[:5]) == [10101, 20201, 110101, 20203, 20101]
    assert (a_data["RECTYPE"].iloc[:5] == 3).all()
    _assert_dtypes_match_ddi(a_data, ddi)


def _assert_atus_00035_df(data: pd.DataFrame, ddi):
    """Run all the checks for the data frame returned by our readers for hierarchical files"""
    assert len(data) == 255885
    assert len(data.columns) == 15
    assert (data["YEAR"].iloc[:5] == 2016).all()
    assert _values(data["RECTYPE"].iloc[:5]) == [1, 2, 3, 3, 3]
    # variables only exist on their own record type; other rows are missing
    assert _values(data["WT06"].iloc[:5]) == [None, 24588650.161504, None, None, None]
    assert _values(data["STATEFIP"].iloc[:5]) == [13, None, None, None, None]
    assert _values(data["ACTLINE"].iloc[:5]) == [None, None, 1, 2, 3]
    assert _values(data["START"].iloc[:5]) == [
        None,
        None,
        "04:00:00",
        "11:00:00",
        "11:20:00",
    ]
    _assert_dtypes_match_ddi(data, ddi, overrides={"WT06": pd.Float64Dtype()})


def _assert_cps_rectantular_subset(data: pd.DataFrame):
    """Tests subset functionality on rectangular extracts"""
    assert len(data.columns) == 2
    assert (data["STATEFIP"].iloc[:5] == np.array([55, 55, 55, 27, 27])).all()


def _assert_cps_hierarchical_subset(data: pd.DataFrame):
    """Tests subset functionality on hierarchical extracts"""
    # even though specified subset is 3 vars, we must add SERIAL
    assert len(data.columns) == 4
    assert "SERIAL" in data.columns
    # there has to be a better way to do this...
    # splitting out nan and non-nan values
    print(data.head())
    assert (data["MISH"].iloc[:2] == np.array([7, 5])).all()
    assert data["MISH"].iloc[2:5].isna().all()
    assert (data["AGE"].iloc[2:5] == np.array([36, 41, 5])).all()
    assert data["AGE"].iloc[:2].isna().all()
    assert (data["RECTYPE"].iloc[:3] == np.array(["H", "H", "P"])).all()


def _assert_cps_hierarchical_subset_dict(data: Dict):
    """Tests subset functionality on hierarchical extracts as dictionaries"""
    p_data = data["P"]
    h_data = data["H"]
    assert len(p_data.columns) == 3
    assert len(h_data.columns) == 3
    assert (h_data["MISH"].iloc[:5] == np.array([7, 5, 1, 2, 1])).all()
    assert (p_data["AGE"].iloc[:5] == np.array([36, 41, 5, 7, 50])).all()


def test_can_read_hierarchical_df_dat_gz(fixtures_path: Path):
    """
    Confirm that we can read hierarchical microdata ino a single data frame
    in .dat format when it is gzipped
    """
    ddi = readers.read_ipums_ddi(fixtures_path / "cps_00421.xml")
    data = readers.read_hierarchical_microdata(
        ddi, fixtures_path / "cps_00421.dat.gz", as_dict=False
    )

    _assert_cps_00421_df(data, ddi)

    ddi = readers.read_ipums_ddi(fixtures_path / "atus_00035.xml")
    data = readers.read_hierarchical_microdata(
        ddi, fixtures_path / "atus_00035.dat.gz", as_dict=False
    )

    _assert_atus_00035_df(data, ddi)

    ddi = readers.read_ipums_ddi(fixtures_path / "meps_00045.xml")
    data = readers.read_hierarchical_microdata(
        ddi, fixtures_path / "meps_00045.dat.gz", as_dict=False
    )

    _assert_meps_00045_df(data, ddi)


def test_can_read_hierarchical_dict_dat_gz(fixtures_path: Path):
    """
    Confirm that we can read hierarchical microdata ino a dictionary of data frames
    in .dat format when it is gzipped
    """
    ddi = readers.read_ipums_ddi(fixtures_path / "cps_00421.xml")
    data = readers.read_hierarchical_microdata(
        ddi, fixtures_path / "cps_00421.dat.gz", as_dict=True
    )

    _assert_cps_00421_dict(data, ddi)

    ddi = readers.read_ipums_ddi(fixtures_path / "atus_00035.xml")
    data = readers.read_hierarchical_microdata(
        ddi, fixtures_path / "atus_00035.dat.gz", as_dict=True
    )

    _assert_atus_00035_dict(data, ddi)

    ddi = readers.read_ipums_ddi(fixtures_path / "meps_00045.xml")
    data = readers.read_hierarchical_microdata(
        ddi, fixtures_path / "meps_00045.dat.gz", as_dict=True
    )

    _assert_meps_00045_dict(data, ddi)


def test_can_read_rectangular_dat_gz(fixtures_path: Path):
    """
    Confirm that we can read rectangular microdata in .dat format
    when it is gzipped
    """
    ddi = readers.read_ipums_ddi(fixtures_path / "cps_00006.xml")
    data = readers.read_microdata(ddi, fixtures_path / "cps_00006.dat.gz")

    _assert_cps_000006(data)


def test_can_read_rectangular_R_dat_gz(fixtures_path: Path):
    """
    Confirm that we can read rectangular on R MEPS microdata in .dat format
    when it is gzipped
    """
    ddi = readers.read_ipums_ddi(fixtures_path / "meps_00019.xml")
    data = readers.read_microdata(ddi, fixtures_path / "meps_00019.dat.gz")

    print(data.dtypes.values)

    _assert_meps_000019(data)


def test_can_read_rectangular_csv_gz(fixtures_path: Path):
    """
    Confirm that we can read rectangular microdata in .csv format
    when it is gzipped
    """
    ddi = readers.read_ipums_ddi(fixtures_path / "cps_00006.xml")
    data = readers.read_microdata(ddi, fixtures_path / "cps_00006.csv.gz")

    _assert_cps_000006(data)


def test_can_read_rectangular_dat(fixtures_path: Path):
    """
    Confirm that we can read rectangular microdata in .dat format
    when it is not gzipped
    """
    ddi = readers.read_ipums_ddi(fixtures_path / "cps_00006.xml")

    ## Un gzip the file in our fixtures
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        with gzip.open(fixtures_path / "cps_00006.dat.gz", "rb") as infile:
            with open(tmpdir / "cps_00006.dat", "wb") as outfile:
                for chunk in iter(partial(infile.read, 8192), b""):
                    outfile.write(chunk)

        data = readers.read_microdata(ddi, tmpdir / "cps_00006.dat")

    _assert_cps_000006(data)


def test_can_read_rectangular_parquet(fixtures_path: Path):
    """
    Confirm that we can read rectangular microdata in .parquet format
    """
    ddi = readers.read_ipums_ddi(fixtures_path / "cps_00006.xml")
    data = readers.read_microdata(ddi, fixtures_path / "cps_00006.parquet")

    _assert_cps_000006(data)


@pytest.mark.slow
def test_can_read_rectangular_dat_gz_chunked(fixtures_path: Path):
    """
    Confirm that we can read rectangular microdata in .dat format when chunked
    """
    ddi = readers.read_ipums_ddi(fixtures_path / "cps_00006.xml")
    data = readers.read_microdata_chunked(
        ddi, fixtures_path / "cps_00006.dat.gz", chunksize=1
    )

    total_length = 0
    seen = 0
    first_hwtsupp = np.array([1475.59, 1475.59, 1475.59, 1597.61, 1706.65])
    for df in data:
        total_length += len(df)
        assert len(df.columns) == 8
        if seen < 5:
            assert (df["YEAR"].loc[: 5 - seen] == 1962).all()
            assert (
                df["HWTSUPP"].iloc[: 5 - seen]
                == first_hwtsupp[seen : min(len(df) + seen, 5)]
            ).all()
            seen += len(df)
    assert total_length == 7668


def test_read_microdata_doubles_numpy(fixtures_path):
    """
    Make sure that fw extracts with float data can be read
    """
    # Checking default behaviour
    numpy_types = {
        "YEAR": np.float64,
        "CASEID": np.float64,
        "SERIAL": np.float64,
        "STATEFIP": np.float64,
        "PERNUM": np.float64,
        "LINENO": np.float64,
        "WT06": np.float64,
        "AGE": np.float64,
        "SEX": np.float64,
        "ACTIVITY": np.float64,
        "START": object,
        "STOP": object,
        "BLS_PCARE": np.float64,
    }

    ddi = readers.read_ipums_ddi(fixtures_path / "atus_00034.xml")
    dtype = ddi.get_all_types(type_format="numpy_type")
    data = readers.read_microdata(ddi, fixtures_path / "atus_00034.dat.gz", dtype=dtype)
    assert data.dtypes.to_dict() == numpy_types


def test_read_microdata_doubles_pandas(fixtures_path):
    """
    Make sure that fw extracts with float data can be read
    """
    pandas_types = {
        "YEAR": pd.Int64Dtype(),
        "CASEID": pd.Int64Dtype(),
        "SERIAL": pd.Int64Dtype(),
        "STATEFIP": pd.Int64Dtype(),
        "PERNUM": pd.Int64Dtype(),
        "LINENO": pd.Int64Dtype(),
        "WT06": pd.Float64Dtype(),
        "AGE": pd.Int64Dtype(),
        "SEX": pd.Int64Dtype(),
        "ACTIVITY": pd.Int64Dtype(),
        "START": pd.StringDtype(),
        "STOP": pd.StringDtype(),
        "BLS_PCARE": pd.Int64Dtype(),
    }

    ddi = readers.read_ipums_ddi(fixtures_path / "atus_00034.xml")
    data = readers.read_microdata(ddi, fixtures_path / "atus_00034.dat.gz")
    assert data.dtypes.to_dict() == pandas_types


def test_read_microdata_doubles_other(fixtures_path):
    """
    Make sure that fw extracts with float data can be read
    """

    pandas_types_other = {
        "YEAR": pd.Int64Dtype(),
        "CASEID": pd.Int64Dtype(),
        "SERIAL": pd.Int64Dtype(),
        "STATEFIP": pd.Int64Dtype(),
        "PERNUM": pd.Int64Dtype(),
        "LINENO": pd.Int64Dtype(),
        "WT06": pd.Float64Dtype(),
        "AGE": pd.Int64Dtype(),
        "SEX": pd.Int64Dtype(),
        "ACTIVITY": pd.Int64Dtype(),
        "START": pd.StringDtype(storage="pyarrow"),
        "STOP": pd.StringDtype(storage="pyarrow"),
        "BLS_PCARE": pd.Int64Dtype(),
    }

    ddi = readers.read_ipums_ddi(fixtures_path / "atus_00034.xml")
    dtype = ddi.get_all_types(type_format="pandas_type", string_pyarrow=True)
    data = readers.read_microdata(ddi, fixtures_path / "atus_00034.dat.gz", dtype=dtype)
    assert data.dtypes.to_dict() == pandas_types_other


def test_read_microdata_custom_dtype(fixtures_path):
    """
    Make sure use can choose custom dtype in microdata reader.
    """
    # Checking default behaviour
    pandas_types = {
        "YEAR": pd.Int64Dtype(),
        "SERIAL": pd.Int64Dtype(),
        "MONTH": pd.Int64Dtype(),
        "HWTFINL": np.float64,
        "CPSID": pd.Int64Dtype(),
        "ASECFLAG": pd.Int64Dtype(),
        "STATEFIP": pd.Int64Dtype(),
        "HRSERSUF": pd.StringDtype(),
        "PERNUM": pd.Int64Dtype(),
        "WTFINL": np.float64,
        "CPSIDP": pd.Int64Dtype(),
        "AGE": pd.Int64Dtype(),
    }

    ddi = readers.read_ipums_ddi(fixtures_path / "cps_00361.xml")
    data = readers.read_microdata(
        ddi, fixtures_path / "cps_00361.dat.gz", dtype=pandas_types
    )
    assert data.dtypes.to_dict() == pandas_types

    # custom dtype
    pandas_types_efficient = {
        "YEAR": np.float64,
        "SERIAL": np.float64,
        "MONTH": np.float64,
        "HWTFINL": np.float64,
        "CPSID": np.float64,
        "ASECFLAG": np.float64,
        "STATEFIP": np.float64,
        "HRSERSUF": pd.StringDtype(storage="pyarrow"),
        "PERNUM": np.float64,
        "WTFINL": np.float64,
        "CPSIDP": np.float64,
        "AGE": np.float64,
    }

    ddi = readers.read_ipums_ddi(fixtures_path / "cps_00361.xml")
    dtype = ddi.get_all_types(type_format="pandas_type_efficient", string_pyarrow=True)
    data = readers.read_microdata(
        ddi, fixtures_path / "cps_00361.dat.gz", dtype=pandas_types_efficient
    )
    assert data.dtypes.to_dict() == pandas_types_efficient

    with pytest.raises(ValueError):
        # should raise Value when parquet and dtype != None
        readers.read_microdata(ddi, fixtures_path / "cps_00006.parquet", dtype=dtype)


def test_read_microdata_chunked_custom_dtype(fixtures_path):
    """
    Make sure use can choose custom dtype in microdata reader.
    """
    # Checking default behaviour
    pandas_types = {
        "YEAR": pd.Int64Dtype(),
        "SERIAL": pd.Int64Dtype(),
        "MONTH": pd.Int64Dtype(),
        "HWTFINL": np.float64,
        "CPSID": pd.Int64Dtype(),
        "ASECFLAG": pd.Int64Dtype(),
        "STATEFIP": pd.Int64Dtype(),
        "HRSERSUF": pd.StringDtype(),
        "PERNUM": pd.Int64Dtype(),
        "WTFINL": np.float64,
        "CPSIDP": pd.Int64Dtype(),
        "AGE": pd.Int64Dtype(),
    }

    ddi = readers.read_ipums_ddi(fixtures_path / "cps_00361.xml")
    data = readers.read_microdata_chunked(
        ddi, fixtures_path / "cps_00361.dat.gz", dtype=pandas_types
    )
    assert next(data).dtypes.to_dict() == pandas_types

    # custom dtype
    pandas_types_efficient = {
        "YEAR": np.float64,
        "SERIAL": np.float64,
        "MONTH": np.float64,
        "HWTFINL": np.float64,
        "CPSID": np.float64,
        "ASECFLAG": np.float64,
        "STATEFIP": np.float64,
        "HRSERSUF": pd.StringDtype(storage="pyarrow"),
        "PERNUM": np.float64,
        "WTFINL": np.float64,
        "CPSIDP": np.float64,
        "AGE": np.float64,
    }

    ddi = readers.read_ipums_ddi(fixtures_path / "cps_00361.xml")
    dtype = ddi.get_all_types(type_format="pandas_type_efficient", string_pyarrow=True)
    data = readers.read_microdata_chunked(
        ddi, fixtures_path / "cps_00361.dat.gz", dtype=dtype
    )
    assert next(data).dtypes.to_dict() == pandas_types_efficient

    with pytest.raises(ValueError):
        # should raise Value when parquet and dtype != None
        next(
            readers.read_microdata_chunked(
                ddi, fixtures_path / "cps_00006.parquet", dtype=dtype
            )
        )


def test_read_extract_description(fixtures_path: Path):
    """
    Make sure that equivalent extracts can be read as either json or yaml and that
    if a badly formatted extract is provided, we raise a ValueError
    """
    yaml_extract = readers.read_extract_description(
        fixtures_path / "example_extract_v2.yml"
    )
    json_extract = readers.read_extract_description(
        fixtures_path / "example_extract_v2.json"
    )
    from_api_extract = readers.read_extract_description(
        fixtures_path / "example_extract_from_api_v2.json"
    )
    from_api_extract_fancy = readers.read_extract_description(
        fixtures_path / "example_fancy_extract_from_api_v2.json"
    )

    # Make sure they are the same
    assert yaml_extract == json_extract

    # Make sure the contents are correct
    assert yaml_extract == {
        "extracts": [
            {
                "description": "Simple IPUMS extract",
                "collection": "usa",
                "version": 2,
                "samples": ["us2012b"],
                "variables": ["AGE", "SEX", "RACE"],
                "dataStructure": {"rectangular": {"on": "P"}},
                "dataFormat": "fixed_width",
            }
        ],
    }

    extract_description = yaml_extract["extracts"][0]
    extract = MicrodataExtract(**extract_description)

    extract_description = from_api_extract["extracts"][0]
    api_extract = MicrodataExtract(**extract_description)

    assert isinstance(extract, MicrodataExtract)
    assert isinstance(api_extract, MicrodataExtract)

    assert extract.build() == api_extract.build()

    # check that this can read fancier things as well
    extract_description_fancy = from_api_extract_fancy["extracts"][0]
    api_extract_fancy = MicrodataExtract(**extract_description_fancy)

    # truncated test
    assert api_extract_fancy.build()["variables"]["AGE"] == {
        "preselected": False,
        "caseSelections": {"general": [1, 2, 3]},
        "attachedCharacteristics": [],
        "dataQualityFlags": False,
        "adjustMonetaryValues": False,
    }

    # Check that something that is neither YAML nor JSON yields a ValueError
    with pytest.raises(ValueError):
        readers.read_extract_description(fixtures_path / "cps_00006.xml")


def test_subset_option(fixtures_path: Path):
    """Does subset option function for all data structures"""
    # first for rectangular
    ddi = readers.read_ipums_ddi(fixtures_path / "cps_00006.xml")
    data = readers.read_microdata(
        ddi, fixtures_path / "cps_00006.dat.gz", subset=["STATEFIP", "INCTOT"]
    )

    _assert_cps_rectantular_subset(data)

    # then for hierarchical single data frame
    ddi = readers.read_ipums_ddi(fixtures_path / "cps_00421.xml")
    data = readers.read_hierarchical_microdata(
        ddi,
        fixtures_path / "cps_00421.dat.gz",
        subset=["RECTYPE", "MISH", "AGE"],
        as_dict=False,
    )

    _assert_cps_hierarchical_subset(data)

    # then for hierarchical dictionary
    data = readers.read_hierarchical_microdata(
        ddi,
        fixtures_path / "cps_00421.dat.gz",
        subset=["RECTYPE", "MISH", "AGE"],
        as_dict=True,
    )

    _assert_cps_hierarchical_subset_dict(data)

    # warn when the rectype var is not included in the subset
    with pytest.warns() as record:
        data = readers.read_hierarchical_microdata(
            ddi, fixtures_path / "cps_00421.dat.gz", subset=["MISH", "AGE"]
        )

    assert (
        str(record[0].message)
        == "RECTYPE is required to read hierarchical microdata; this variable has been added to the `subset`."
    )
