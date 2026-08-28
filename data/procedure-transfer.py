import pandas as pd

df = pd.read_csv("hospitals-final-cleaned.csv", dtype=str)

print("COLUMNS:")
print(df.columns.tolist())

print("\nSPECIALTY VALUES CONTAINING 'SURGERY':")

for value in df["Specialties"].dropna():
    for item in str(value).split(";"):
        if "surg" in item.lower():
            print(repr(item.strip()))
