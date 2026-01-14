from dataclasses import dataclass
from pathlib import Path
from typing import List
import pandas as pd

REQUIRED_COLUMNS = {"company_name","reference_name", "careers_url"}

@dataclass
class Company:
    company_name: str
    reference_name: str
    careers_url: str

def load_companies(excel_path:str)-> List[Company]:
    project_root = Path(__file__).resolve().parents[1]
    path = project_root / excel_path

    if not path.exists():
        raise FileNotFoundError(f"Excel file not found at - {path}")

    df = pd.read_excel(path)

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(f"Following columns are missing in file {missing}")

    # drop rows with missing values in carrer_url column
    df = df.dropna(subset = ["careers_url"])

    companies : List[Company] = []

    for _,row in df.iterrows():
        company = Company(
            company_name = str(row['company_name']).strip(),
            reference_name = str(row["reference_name"]).strip(),
            careers_url= str(row["careers_url"]).strip()
        )

        if company.careers_url:
            companies.append(company)

    return companies