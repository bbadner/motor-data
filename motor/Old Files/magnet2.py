import os
import re
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import fitz
from PIL import Image, ImageOps
import pytesseract

# -------------------------------
# Tesseract location
# -------------------------------

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\bbadner\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# -------------------------------
# CONFIG
# -------------------------------

TARGET = 3.7
TOL = 0.5

LSL = TARGET - TOL
USL = TARGET + TOL

TARGET_VALUES = 30
MIN_VALUES = 10

REPORT_FOLDER = "Magnet Reports"
os.makedirs(REPORT_FOLDER, exist_ok=True)

# -------------------------------
# Cp/Cpk calculation
# -------------------------------

def compute_cp_cpk(mean, std):

    if std <= 0:
        return 0, 0

    cp = (USL - LSL) / (6 * std)

    cpk = min(
        (mean - LSL) / (3 * std),
        (USL - mean) / (3 * std)
    )

    return cp, cpk


# -------------------------------
# Render PDF page to image
# -------------------------------

def render_page(pdf, page):

    doc = fitz.open(pdf)

    mat = fitz.Matrix(4, 4)

    pix = doc.load_page(page).get_pixmap(matrix=mat)

    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    doc.close()

    return img


# -------------------------------
# OCR helper
# -------------------------------

def ocr(img):

    gray = img.convert("L")

    gray = ImageOps.autocontrast(gray)

    text = pytesseract.image_to_string(gray)

    return text


# -------------------------------
# Extract month
# -------------------------------

def extract_month(text, filename):

    m = re.search(r'([A-Za-z]{3})/(\d{1,2})/(\d{4})', text)

    if m:
        month = datetime.strptime(m.group(1), "%b").month
        return datetime(int(m.group(3)), month, 1)

    m = re.search(r'([A-Za-z]{3})-(\d{1,2})-(\d{4})', text)

    if m:
        month = datetime.strptime(m.group(1), "%b").month
        return datetime(int(m.group(3)), month, 1)

    m = re.search(r'(20\d{2})[-_](\d{2})[-_](\d{2})', filename)

    if m:
        return datetime(int(m.group(1)), int(m.group(2)), 1)

    return None


# -------------------------------
# Extract values
# -------------------------------

def extract_values(text):

    matches = re.findall(r'3[\.\,\:\s]\d{2}', text)

    values = []

    for m in matches:

        v = m.replace(",", ".").replace(":", ".").replace(" ", ".")

        try:

            v = float(v)

            if 3.0 <= v <= 4.5:
                values.append(v)

        except:
            pass

    return values[:TARGET_VALUES]


# -------------------------------
# Extract angle values
# -------------------------------

def extract_angle_values(pdf):

    img = render_page(pdf, 1)

    text = ocr(img)

    values = extract_values(text)

    # fallback if OCR struggles
    if len(values) < 20:

        img = img.rotate(90, expand=True)

        text = ocr(img)

        values = extract_values(text)

    return values


# -------------------------------
# Create report
# -------------------------------

def create_report(rows):

    labels = [r["label"] for r in rows]
    cpks = [r["cpk"] for r in rows]

    plt.figure(figsize=(8, 4))

    plt.plot(labels, cpks, marker="o")

    plt.axhline(1.33, linestyle="--")

    plt.title("Magnet Timing CpK Trend")

    plt.ylabel("Cpk")

    plt.grid(True)

    chart = "chart.png"

    plt.tight_layout()

    plt.savefig(chart)

    plt.close()

    pdf = FPDF()

    pdf.add_page()

    pdf.set_font("Arial", "B", 16)

    pdf.cell(0, 10, "Rexair Magnet Timing CpK Report", ln=True)

    pdf.image(chart, x=10, y=30, w=180)

    pdf.set_y(120)

    pdf.set_font("Arial", "", 11)

    for r in rows:

        line = f"{r['label']}   Mean:{r['mean']:.4f}   Std:{r['std']:.4f}   Cp:{r['cp']:.2f}   Cpk:{r['cpk']:.2f}"

        pdf.cell(0, 8, line, ln=True)

    output = os.path.join(
        REPORT_FOLDER,
        f"Rexair_Magnet_Timing_CpK_Report_{datetime.now():%Y-%m}.pdf"
    )

    pdf.output(output)

    os.remove(chart)

    return output


# -------------------------------
# MAIN
# -------------------------------

def run():

    rows = []

    for file in os.listdir():

        if not file.lower().endswith(".pdf"):
            continue

        if file.startswith("Rexair_Magnet_Timing"):
            continue

        print("\nProcessing:", file)

        doc = fitz.open(file)

        if doc.page_count < 2:

            print("Skipped - not enough pages")

            continue

        page1 = doc.load_page(0).get_text()

        if len(page1.strip()) < 20:

            img = render_page(file, 0)

            page1 = ocr(img)

        month = extract_month(page1, file)

        if not month:

            print("Skipped - month not detected")

            continue

        values = extract_angle_values(file)

        if len(values) < MIN_VALUES:

            print("Skipped - insufficient values:", len(values))

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

        print(f"OK n={len(values)} mean={mean:.4f} std={std:.4f} cpk={cpk:.2f}")

    rows.sort(key=lambda r: r["month_dt"])

    for r in rows:

        r["label"] = r["month_dt"].strftime("%b %Y")

    output = create_report(rows)

    print("\nReport created:")
    print(output)


if __name__ == "__main__":
    run()