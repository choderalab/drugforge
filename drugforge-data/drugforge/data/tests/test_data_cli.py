import json

import pandas
import pytest
from drugforge.data.cli.cli import data as cli
from drugforge.data.testing.test_resources import fetch_test_file
from click.testing import CliRunner


@pytest.fixture(scope="session")
def cdd_to_schema_files():
    in_fn = fetch_test_file("test_cdd_to_schema_in.csv")

    out_json_fn = fetch_test_file("test_cdd_to_schema_out.json")
    out_csv_fn = fetch_test_file("test_cdd_to_schema_out.csv")

    return in_fn, out_csv_fn, out_json_fn


def test_cdd_to_schema(cdd_to_schema_files, tmp_path):
    in_fn, out_csv_fn, out_json_fn = cdd_to_schema_files

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "cdd-to-schema",
            "--in-file",
            in_fn,
            "--out-json",
            tmp_path / "test_cdd_to_schema_out.json",
            "--out-csv",
            tmp_path / "test_cdd_to_schema_out.csv",
        ],
    )
    assert result.exit_code == 0

    # Make sure files exist
    test_out_json_fn = tmp_path / "test_cdd_to_schema_out.json"
    test_out_csv_fn = tmp_path / "test_cdd_to_schema_out.csv"
    assert test_out_json_fn.exists()
    assert test_out_csv_fn.exists()

    # Check files are right
    df_check = pandas.read_csv(out_csv_fn, index_col=0)
    df_test = pandas.read_csv(test_out_csv_fn, index_col=0)
    assert df_test.equals(df_check)

    json_check = json.loads(out_json_fn.read_text())
    json_test = json.loads(test_out_json_fn.read_text())

    json_check = sorted(json_check, key=lambda d: d.get("compound_id"))
    json_test = sorted(json_test, key=lambda d: d.get("compound_id"))

    # note in one case, the stdevs in the experimental data
    # are listed as nan, in the other as None
    # e.g.,: 'dG_stderr': nan, 'dG_95ci_lower': nan, 'dG_95ci_upper': nan
    # nan and None are not equal, obviously, in this comparison, but are effectively equivalent in their meaning here
    # so we just need to do some extra work to ensure that those specific values are treated as equal
    # This is just a recursive function that will loop through the entries.
    # If the entry is not equivalent, it will dig down into the individual elements or subelements
    # It will also treat nan and None as equivalent
    import math
    def compare_vals_in_dict(d1, d2):
        if d1 == d2:
            return True
        elif isinstance(d1, float) or isinstance(d2, float):
            if isinstance(d1, float) and math.isnan(d1) and d2 is None:
                return True
            elif isinstance(d2, float) and math.isnan(d2) and d1 is None:
                return True
            else:
                return False
        elif isinstance(d1, dict) and isinstance(d2, dict):
            for key in d1.keys():
                if not compare_vals_in_dict(d1[key], d2[key]):
                    return False
            return True

    equalivance = []
    for d1, d2 in zip(json_check, json_test):
        equalivance.append(compare_vals_in_dict(d1, d2))

    assert all(equalivance)
