import os
import fcsparser
from pathlib import Path

def fcs_to_csv(directory):
    path = Path('.')
    directory_fcs = path / directory / "FCS"
    directory_csv = path / directory / "CSV"
    os.makedirs(directory_csv, exist_ok=True)
    all_files = os.listdir(directory_fcs)
    for fcs_file in all_files:
        fcs_path = directory_fcs / fcs_file
        try: 
            metadata, data = fcsparser.parse(fcs_path)

            csv_file = Path(fcs_path).stem # Gets the file name without extension
            csv_file = f"{csv_file}.csv"
            csv_path = directory_csv / csv_file
            data.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"Error {fcs_file}: {e}")

fcs_to_csv("BLAST110")
fcs_to_csv("LAIP29")