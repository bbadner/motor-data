import os
import re
import pdfplumber
from datetime import datetime
from collections import defaultdict

# ===============================
# SETTINGS
# ===============================

FOLDER = os.getcwd()
REPORT_FILE = "magnet_summary_report.txt"

# ===============================
# FUNCTIONS
# ===============================

def extract_inspection_date(text):
    """
    Extract inspection date from page 1
    Example: Apr/14/2025
    """
    match = re.search(r'Inspection\s*[:：]?\s*([A-Za-z]{3}/\d{1,2}/\d{4})', text, re.IGNORECASE)
    if not match:
        return None

    try:
        dt = datetime.strptime(match.group(1), "%b/%d/%Y")
        return dt.strftime("%b %Y")
    except:
        return None


def extract_angle_data(text):
    """
    Extract only the 3.7 column values
    """
    values = []

    lines = text.split("\n")

    for line in lines:
        match = re.findall(r'(\d+\.\d+)°', line)
        if match:
            values.extend([float(x) for x in match])

    return values


def process_pdf(path):
    """
    Process a single magnet report
    """
    try:
        with pdfplumber.open(path) as pdf:

            if len(pdf.pages) < 2:
                return None, None, "INCOMPLETE"

            page1 = pdf.pages[0].extract_text()
            page2 = pdf.pages[1].extract_text()

            date = extract_inspection_date(page1)

            if not date:
                return None, None, "INCOMPLETE"

            data = extract_angle_data(page2)

            if len(data) == 0:
                return None, None, "INCOMPLETE"

            return date, data, "OK"

    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None, None, "INCOMPLETE"


# ===============================
# MAIN PROCESS
# ===============================

month_data = defaultdict(list)
duplicates = []
incomplete_files = []

print("Scanning PDFs...")

for file in os.listdir(FOLDER):

    if not file.lower().endswith(".pdf"):
        continue

    path = os.path.join(FOLDER, file)

    date, data, status = process_pdf(path)

    if status == "INCOMPLETE":
        incomplete_files.append(file)
        continue

    # Duplicate detection
    duplicate_found = False

    for existing in month_data[date]:
        if existing["values"] == data:
            duplicates.append((file, existing["file"]))
            duplicate_found = True
            break

    if not duplicate_found:
        month_data[date].append({
            "file": file,
            "values": data
        })


# ===============================
# REPORT GENERATION
# ===============================

report_lines = []
report_lines.append("MAGNET REPORT SUMMARY\n")
report_lines.append("="*50 + "\n")

for month in sorted(month_data.keys()):

    report_lines.append(f"\n{month}\n")
    report_lines.append("-"*30 + "\n")

    for entry in month_data[month]:

        values = entry["values"]

        mean = sum(values)/len(values)
        minimum = min(values)
        maximum = max(values)

        report_lines.append(f"File: {entry['file']}\n")
        report_lines.append(f"Samples: {len(values)}\n")
        report_lines.append(f"Mean: {mean:.3f}\n")
        report_lines.append(f"Min: {minimum:.3f}\n")
        report_lines.append(f"Max: {maximum:.3f}\n\n")


# ===============================
# FLAGS
# ===============================

if incomplete_files:

    report_lines.append("\n\nINCOMPLETE REPORT RECEIVED\n")

    for f in incomplete_files:
        report_lines.append(f"{f}\n")


if duplicates:

    report_lines.append("\n\nDUPLICATE REPORTS INCLUDED\n")

    for d in duplicates:
        report_lines.append(f"{d[0]} duplicates {d[1]}\n")


# ===============================
# SAVE REPORT
# ===============================
with open(REPORT_FILE, "w", encoding="utf-8") as f:
#with open(REPORT_FILE, "w") as f:
    f.writelines(report_lines)

print("\nReport created:")
print(REPORT_FILE)