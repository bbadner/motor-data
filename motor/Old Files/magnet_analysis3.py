# ============================================================
# Rexair Magnet Timing Analysis
# Reliable extractor for supplier magnet timing reports
# ============================================================

import os
import re
from datetime import datetime
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import fitz

REPORT_FOLDER = "Magnet Reports"

TARGET = 3.7
TOL = 0.5
LSL = TARGET - TOL
USL = TARGET + TOL

TARGET_VALUES = 30
MIN_VALUES = 15

os.makedirs(REPORT_FOLDER, exist_ok=True)


def compute_cp_cpk(mean, std):
    if std <= 0:
        return 0, 0
    cp = (USL - LSL) / (6 * std)
    cpk = min((mean - LSL) / (3 * std), (USL - mean) / (3 * std))
    return cp, cpk


def extract_month(page1_text, filename):

    patterns = [
        r'Inspection\s*[:：]?\s*([A-Za-z]{3}/\d{1,2}/\d{4})',
        r'INSPECTION\s*[:：]?\s*([A-Za-z]{3}/\d{1,2}/\d{4})'
    ]

    for pat in patterns:
        m = re.search(pat, page1_text, re.IGNORECASE)
        if m:
            try:
                return datetime.strptime(m.group(1), "%b/%d/%Y").replace(day=1)
            except:
                pass

    m = re.search(r'(20\d{2})[-_](\d{2})[-_](\d{2})', filename)
    if m:
        return datetime(int(m.group(1)), int(m.group(2)), 1)

    return None


def extract_37_values(page2_text):

    lines = page2_text.splitlines()
    values = []

    for line in lines:

        if not re.match(r"^\d+", line.strip()):
            continue

        nums = re.findall(r"([34]\.\d{1,2})", line)

        if nums:
            v = float(nums[-1])  # take right-most number
            if 3.0 <= v <= 4.5:
                values.append(v)

    if len(values) >= TARGET_VALUES:
        return values[-TARGET_VALUES:]

    return values


def create_pdf(output_path, rows):

    labels = [r["label"] for r in rows]
    cpks = [r["cpk"] for r in rows]

    plt.figure(figsize=(8,4))
    plt.plot(labels, cpks, marker="o")
    plt.axhline(1.33, linestyle="--")
    plt.title("Magnet Timing CpK Trend")
    plt.ylabel("CpK")
    plt.grid(True)

    chart = "magnet_chart.png"
    plt.tight_layout()
    plt.savefig(chart)
    plt.close()

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial","B",16)
    pdf.cell(0,10,"Rexair Magnet Timing CpK Report",ln=True)

    pdf.image(chart, x=10, y=30, w=180)

    pdf.set_y(120)
    pdf.set_font("Arial","B",12)
    pdf.cell(0,10,"Per-File Summary",ln=True)

    pdf.set_font("Arial","",11)

    for r in rows:
        pdf.cell(
            0,8,
            f"{r['label']} Mean:{r['mean']:.4f} Std:{r['std']:.4f} Cp:{r['cp']:.2f} CpK:{r['cpk']:.2f}",
            ln=True
        )

    pdf.output(output_path)

    if os.path.exists(chart):
        os.remove(chart)


def run():

    base = os.getcwd()
    rows = []

    for file in os.listdir(base):

        if not file.lower().endswith(".pdf"):
            continue

        path = os.path.join(base, file)

        try:
            doc = fitz.open(path)
        except:
            continue

        if doc.page_count < 2:
            continue

        page1 = doc.load_page(0).get_text()
        page2 = doc.load_page(1).get_text()

        month = extract_month(page1, file)

        if not month:
            continue

        values = extract_37_values(page2)

        if len(values) < MIN_VALUES:
            continue

        arr = np.array(values)

        mean = arr.mean()
        std = arr.std(ddof=1)

        cp, cpk = compute_cp_cpk(mean, std)

        rows.append({
            "month_dt": month,
            "mean": mean,
            "std": std,
            "cp": cp,
            "cpk": cpk,
            "file": file
        })

        print(f"[MAGNET] OK: {file} n={len(values)} mean={mean:.4f}")

    rows.sort(key=lambda r: r["month_dt"])

    month_count = defaultdict(int)

    for r in rows:
        month_count[r["month_dt"]] += 1
        r["label"] = f"{r['month_dt'].strftime('%B')} {month_count[r['month_dt']]}"

    output = os.path.join(
        base,
        REPORT_FOLDER,
        f"Rexair_Magnet_Timing_CpK_Report_{datetime.now():%Y-%m}.pdf"
    )

    create_pdf(output, rows)

    print("\nReport created:")
    print(output)


if __name__ == "__main__":
    run()