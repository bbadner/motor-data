import pandas as pd

# 👉 Change this path if your file is in a different folder
file_path = r"C:\Python Projects\Data Agent\MOTOR TEST DATA\Motor Test Data-IN24021601(C6521240205-0001~6582).xls"

print("Reading file:", file_path)

# Read first 30 rows from the Excel file
df = pd.read_excel(file_path, header=None, nrows=30, engine="xlrd")

print("\n=== Preview of first 30 rows ===")
print(df)
