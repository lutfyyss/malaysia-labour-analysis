from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
CLEAN = Path("data/clean")
CLEAN.mkdir(parents=True, exist_ok=True)


def load(name):
    df = pd.read_csv(RAW / name, parse_dates=["date"])
    n_dup = df.duplicated().sum()
    if n_dup:
        print(f"{name}: dropped {n_dup} duplicate rows")
    return df.drop_duplicates().sort_values("date").reset_index(drop=True)


def add_time_cols(df):
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df


def clean_month():
    df = load("lfs_month.csv")
    assert df["date"].is_unique, "duplicate dates in lfs_month"
    # sanity checks (values are in thousands, rates in %)
    gap = (df["lf"] - df["lf_employed"] - df["lf_unemployed"]).abs()
    print("lfs_month: rows where lf != employed + unemployed (>1k):", (gap > 1).sum())
    calc = df["lf_unemployed"] / df["lf"] * 100
    print("lfs_month: rows where u_rate differs from calculated (>0.2):",
          ((calc - df["u_rate"]).abs() > 0.2).sum())
    print("lfs_month: missing values:\n", df.isna().sum()[df.isna().sum() > 0])
    return add_time_cols(df)


def clean_youth():
    df = load("lfs_month_youth.csv")
    assert df["date"].is_unique, "duplicate dates in lfs_month_youth"
    print("lfs_month_youth: missing values:\n", df.isna().sum()[df.isna().sum() > 0])
    return add_time_cols(df)


def clean_sru():
    df = load("lfs_qtr_sru_age.csv")
    print("lfs_qtr_sru_age variables:", df["variable"].unique())
    print("lfs_qtr_sru_age ages:", df["age"].unique())
    wide = (
        df.pivot_table(index=["date", "age"], columns="variable", values="sru")
        .reset_index()
    )
    wide.columns.name = None
    wide["year"] = wide["date"].dt.year
    wide["quarter"] = wide["date"].dt.quarter
    return wide


def clean_qtr_state():
    df = load("lfs_qtr_state.csv")
    df["state"] = df["state"].str.strip()
    assert not df.duplicated(["date", "state"]).any(), "duplicate date/state"
    print("lfs_qtr_state: states:", df["state"].nunique(),
          "| quarters:", df["date"].nunique())
    gap = (df["lf"] - df["lf_employed"] - df["lf_unemployed"]).abs()
    print("lfs_qtr_state: rows where lf != employed + unemployed (>1k):", (gap > 1).sum())
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    return df


def clean_state_sex():
    df = load("lfs_state_sex.csv")
    df["state"] = df["state"].str.strip()
    df["sex"] = df["sex"].str.strip().str.lower()
    assert not df.duplicated(["date", "state", "sex"]).any(), "duplicate date/state/sex"
    print("lfs_state_sex: sex values:", df["sex"].unique())
    print("lfs_state_sex: first year per state:\n",
          df.groupby("state")["date"].min().dt.year.sort_values().to_string())
    df["year"] = df["date"].dt.year
    return df


if __name__ == "__main__":
    clean_month().to_parquet(CLEAN / "lfs_month.parquet", index=False)
    clean_youth().to_parquet(CLEAN / "lfs_month_youth.parquet", index=False)
    clean_sru().to_parquet(CLEAN / "lfs_qtr_sru_age.parquet", index=False)
    qs = clean_qtr_state()
    ss = clean_state_sex()
    qs.to_parquet(CLEAN / "lfs_qtr_state.parquet", index=False)
    ss.to_parquet(CLEAN / "lfs_state_sex.parquet", index=False)
    # state names must match across the two state files
    print("State names only in one file:",
          set(qs["state"]).symmetric_difference(set(ss["state"])))
    print("Done. Files saved in data/clean")