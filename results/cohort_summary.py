"""
Print cohort statistics for Table A.3 from both eICU and MIMIC-IV.

eICU search locations:
1) EICU_PATH env var (if set)
2) ./eicu
3) <project_root>/eicu

MIMIC-IV search locations:
1) MIMIC_IV_PATH env var (if set)
2) ./mimic-iv
3) <project_root>/mimic-iv
"""

from pathlib import Path
import os

import pandas as pd


def _candidate_dirs(env_var: str, folder_name: str, project_root: Path) -> list[Path]:
    candidates = []
    env_path = os.getenv(env_var)
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend([Path.cwd() / folder_name, project_root / folder_name])

    # Keep order while removing duplicates
    seen = set()
    ordered = []
    for path in candidates:
        norm = str(path.resolve()) if path.exists() else str(path)
        if norm not in seen:
            seen.add(norm)
            ordered.append(path)
    return ordered


def _resolve_table_file(base_dir: Path, rel_no_ext: str) -> Path | None:
    for suffix in (".csv", ".csv.gz"):
        candidate = base_dir / f"{rel_no_ext}{suffix}"
        if candidate.exists():
            return candidate
    return None


def resolve_eicu_patient_file(project_root: Path) -> Path:
    candidate_dirs = _candidate_dirs("EICU_PATH", "eicu", project_root)

    for eicu_dir in candidate_dirs:
        patient_file = _resolve_table_file(eicu_dir, "patient")
        if patient_file is not None:
            return patient_file

    searched = [str(path.resolve()) if path.exists() else str(path) for path in candidate_dirs]
    raise FileNotFoundError(
        "Could not find eICU patient table. Searched:\n- "
        + "\n- ".join(searched)
        + "\nExpected file: patient.csv or patient.csv.gz"
    )


def resolve_mimic_files(project_root: Path) -> dict[str, Path]:
    candidate_roots = _candidate_dirs("MIMIC_IV_PATH", "mimic-iv", project_root)
    required = {
        "patients": "hosp/patients",
        "admissions": "hosp/admissions",
        "icustays": "icu/icustays",
    }

    for root in candidate_roots:
        resolved = {}
        all_found = True
        for key, rel in required.items():
            table_file = _resolve_table_file(root, rel)
            if table_file is None:
                all_found = False
                break
            resolved[key] = table_file
        if all_found:
            return resolved

    searched = [str(path.resolve()) if path.exists() else str(path) for path in candidate_roots]
    raise FileNotFoundError(
        "Could not find required MIMIC-IV tables. Searched roots:\n- "
        + "\n- ".join(searched)
        + "\nRequired: hosp/patients(.csv|.csv.gz), hosp/admissions(.csv|.csv.gz), "
        + "icu/icustays(.csv|.csv.gz)"
    )


def _print_summary(dataset_name: str, data_source: str, cohort: pd.DataFrame, male_label: str) -> None:
    print(f"\n=== {dataset_name} Cohort Statistics for Table A.3 ===")
    print(f"Data source: {data_source}")
    print(f"Total stays: {len(cohort):,}")

    if "hospitalid" in cohort.columns:
        print(f"Unique hospitals: {cohort['hospitalid'].nunique():,}")
    elif dataset_name == "MIMIC-IV":
        print("Unique hospitals: 1")

    age_med = cohort["age_num"].median()
    age_q1 = cohort["age_num"].quantile(0.25)
    age_q3 = cohort["age_num"].quantile(0.75)
    print(f"Age median (IQR): {age_med:.0f} ({age_q1:.0f}-{age_q3:.0f})")

    male_pct = (cohort[male_label].astype(str).str.upper().isin(["MALE", "M"])).mean() * 100
    print(f"Male sex %: {male_pct:.1f}%")

    if "mortality" in cohort.columns:
        mort_pct = cohort["mortality"].mean() * 100
        print(f"In-hospital mortality: {mort_pct:.1f}%")


def summarize_eicu(project_root: Path) -> None:
    patient_file = resolve_eicu_patient_file(project_root)
    pat = pd.read_csv(patient_file, low_memory=False)

    age_clean = (
        pat["age"]
        .astype(str)
        .str.replace("+", "", regex=False)
        .str.replace(">", "", regex=False)
        .str.strip()
    )
    pat["age_num"] = pd.to_numeric(age_clean, errors="coerce")
    pat = pat[pat["age_num"] >= 18].copy()

    if "unitdischargestatus" in pat.columns:
        pat["mortality"] = (pat["unitdischargestatus"].astype(str).str.lower() == "expired").astype(int)
    elif "hospitaldischargestatus" in pat.columns:
        pat["mortality"] = (pat["hospitaldischargestatus"].astype(str).str.lower() == "expired").astype(int)

    _print_summary("eICU", str(patient_file), pat, "gender")


def summarize_mimic(project_root: Path) -> None:
    files = resolve_mimic_files(project_root)

    patients = pd.read_csv(
        files["patients"],
        usecols=["subject_id", "gender", "anchor_age"],
        low_memory=False,
    )
    admissions = pd.read_csv(
        files["admissions"],
        usecols=["subject_id", "hadm_id", "hospital_expire_flag"],
        low_memory=False,
    )
    icustays = pd.read_csv(
        files["icustays"],
        usecols=["subject_id", "hadm_id", "stay_id"],
        low_memory=False,
    )

    cohort = icustays.merge(admissions, on=["subject_id", "hadm_id"], how="inner")
    cohort = cohort.merge(patients, on="subject_id", how="inner")
    cohort["age_num"] = pd.to_numeric(cohort["anchor_age"], errors="coerce")
    cohort = cohort[cohort["age_num"] >= 18].copy()
    cohort["mortality"] = pd.to_numeric(cohort["hospital_expire_flag"], errors="coerce")

    data_source = (
        f"patients={files['patients']}; admissions={files['admissions']}; "
        f"icustays={files['icustays']}"
    )
    _print_summary("MIMIC-IV", data_source, cohort, "gender")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    summarize_eicu(project_root)
    summarize_mimic(project_root)
    print("\nPaste these numbers into Table A.3 of juco_main.tex.")


if __name__ == "__main__":
    main()
