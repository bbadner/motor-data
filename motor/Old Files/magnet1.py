import os
import re
from datetime import datetime
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import fitz
from PIL import Image
import pytesseract

# Path to Tesseract OCR
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\bbadner\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

REPORT_FOLDER = "Magnet Reports"

TARGET = 3.7
TOL = 0.5
LSL = TARGET - TOL
USL = TARGET + TOL

TARGET_VALUES = 30
MIN_VALUES = 10

os.makedirs(REPORT_FOLDER, exist_ok=True)


def compute_cp_cpk(mean, std):
    if std <= 0:
        return 0, 0
    cp = (USL - LSL) / (6 * std)
    cpk = min((mean - LSL) / (3 * std), (USL - mean) / (3 * std))
    return cp, cpk


def extract_month(text, filename):

    # Try Sep/25/2025 style
    m = re.search(r'([A-Za-z]{3})[\.]?\s*(\d{1,2})[\/\-\s](\d{4})', text)

    if m:
        month = datetime.strptime(m.group(1), "%b").month
        year = int(m.group(3))
        return datetime(year, month, 1)

    # Try ISO date in filename
    m = re.search(r'(20\d{2})[-_](\d{2})[-_](\d{2})', filename)

    if m:
        return datetime(int(m.group(1)), int(m.group(2)), 1)

    return None

def extract_values(text):

    values = []

    # find any number that looks like 3.xx or 4.xx
    matches = re.findall(r'[34][\.,]\d{1,2}', text)

    for m in matches:

        v = float(m.replace(",", "."))

        if 3.0 <= v <= 4.5:
            values.append(v)

    # keep first 30 measurements
    if len(values) > 30:
        values = values[:30]

    return values

def ocr_page(pdf_file):

    doc = fitz.open(pdf_file)

    page = doc.load_page(1)

    pix = page.get_pixmap(dpi=400)

    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    text = pytesseract.image_to_string(img)

    doc.close()

    return text


def create_report(rows):

    labels = [r["label"] for r in rows]
    cpks = [r["cpk"] for r in rows]

    plt.figure(figsize=(8,4))
    plt.plot(labels, cpks, marker="o")
    plt.axhline(1.33, linestyle="--")
    plt.title("Magnet Timing CpK Trend")
    plt.ylabel("CpK")
    plt.grid(True)

    chart = "chart.png"

    plt.tight_layout()
    plt.savefig(chart)
    plt.close()

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial","B",16)
    pdf.cell(0,10,"Rexair Magnet Timing CpK Report",ln=True)

    pdf.image(chart, x=10, y=30, w=180)

    pdf.set_y(120)
    pdf.set_font("Arial","",11)

    for r in rows:

        line = f"{r['label']}   Mean:{r['mean']:.4f}   Std:{r['std']:.4f}   Cp:{r['cp']:.2f}   Cpk:{r['cpk']:.2f}"

        pdf.cell(0,8,line,ln=True)

    output = os.path.join(
        REPORT_FOLDER,
        f"Rexair_Magnet_Timing_CpK_Report_{datetime.now():%Y-%m}.pdf"
    )

    pdf.output(output)

    os.remove(chart)

    return output


def run():

    rows = []

    for file in os.listdir():

        if not file.lower().endswith(".pdf"):
            continue

        doc = fitz.open(file)

        if doc.page_count < 2:
            continue

        page1 = doc.load_page(0).get_text()

        # try normal text extraction
        page2_text = doc.load_page(1).get_text()

        # run OCR also
        page2_ocr = ocr_page(file)

        # combine both sources
        page2 = page2_text + "\n" + page2_ocr

        #page1 = doc.load_page(0).get_text()
        #page2 = doc.load_page(1).get_text()

        #if len(page2.strip()) < 20:
            #page2 = ocr_page(file)

        month = extract_month(page1, file)

        if not month:
            continue

        values = extract_values(page2)

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
            "cpk": cpk
        })

        print(f"[MAGNET] OK: {file}  n={len(values)}  mean={mean:.4f}")

    rows.sort(key=lambda r: r["month_dt"])

    counts = defaultdict(int)

    for r in rows:

        counts[r["month_dt"]] += 1

        r["label"] = r["month_dt"].strftime("%b %Y")

    output = create_report(rows)

    print("")
    print("Report created:")
    print(output)


if __name__ == "__main__":
    run()