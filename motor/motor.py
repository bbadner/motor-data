# ============================================================
# Motor & Magnet Analysis - Unified Script
# Motor analysis: per-file, last 12 files, deduplication
# Magnet analysis: full OCR pipeline from mag.py
# ============================================================

import os
import re
import sys
import platform
import subprocess
from datetime import datetime
from collections import defaultdict
from typing import List, Optional, Tuple
import threading
import queue

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fpdf import FPDF
import tkinter as tk
from tkinter import ttk
import fitz  # PyMuPDF

from PIL import Image, ImageOps, ImageFilter
import pytesseract

# ============================================================
# CONFIG
# ============================================================

# Set this to your installed Tesseract path

import shutil

tesseract_path = shutil.which("tesseract")

if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    print("[ERROR] Tesseract not found. Please install Tesseract OCR.")

REPORT_FOLDER_MOTOR  = "Motor Reports"
REPORT_FOLDER_MAGNET = "Magnet Reports"
LOGO_FILE = "rexair_logo.png"

# Magnet spec limits
MAGNET_TARGET = 3.7
MAGNET_TOL    = 0.5
MAG_LSL       = MAGNET_TARGET - MAGNET_TOL   # 3.2
MAG_USL       = MAGNET_TARGET + MAGNET_TOL   # 4.2
TARGET_VALUES = 30
MIN_VALUES    = 10
CHART_TARGET_CPK = 1.33

os.makedirs(REPORT_FOLDER_MOTOR,  exist_ok=True)
os.makedirs(REPORT_FOLDER_MAGNET, exist_ok=True)


# ============================================================
# SECTION 1 — Shared Utilities
# ============================================================

def open_file(filepath):
    """Open a file with the OS default application."""
    try:
        if platform.system() == "Windows":
            os.startfile(filepath)
        elif platform.system() == "Darwin":
            subprocess.run(["open", filepath])
        else:
            subprocess.run(["xdg-open", filepath])
    except Exception as e:
        print(f"[WARN] Could not open file: {e}")


def safe_numeric_array(values):
    s = pd.to_numeric(pd.Series(list(values)), errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s.to_numpy(dtype=float)

def safe_mean(values):
    arr = safe_numeric_array(values)
    return float(arr.mean()) if arr.size else 0.0

def safe_median(values):
    arr = safe_numeric_array(values)
    return float(np.median(arr)) if arr.size else 0.0

def safe_stdev(values):
    arr = safe_numeric_array(values)
    return float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0


# ============================================================
# SECTION 2 — Motor: Data Extraction
# ============================================================

def extract_date_from_filename(fname):
    """Extract date from filenames containing 'C6521YYMMDD'."""
    m = re.search(r"C6521(\d{6})", fname)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%y%m%d")
    except Exception:
        return None

def find_input_power_column(df):
    for i in range(min(10, len(df))):
        row = df.iloc[i].tolist()
        for j, cell in enumerate(row):
            cell_str = str(cell).lower()
            if "high speed(open)" in cell_str and "input power" in cell_str:
                return j

    for i in range(min(10, len(df))):
        row = df.iloc[i].tolist()
        for j, cell in enumerate(row):
            cell_str = str(cell).lower()
            if re.search(r"input\s*power", cell_str):
                return j

    return None

def read_input_power(filepath):
    """Read motor input power values from an Excel file (.xls or .xlsx)."""
    ext    = os.path.splitext(filepath)[1].lower()
    engine = "xlrd" if ext == ".xls" else "openpyxl"
    try:
        df = pd.read_excel(filepath, header=None, engine=engine)
    except Exception as e:
        print(f"[ERROR] Could not read excel: {filepath} -> {e}")
        return []
    col_idx = find_input_power_column(df)
    if col_idx is None:
        print(f"[WARN] 'Input Power' column not found in {os.path.basename(filepath)}")
        return []
    values = []
    for v in df.iloc[:, col_idx].tolist():
        try:
            fv = float(v)
            if np.isfinite(fv):
                values.append(fv)
        except Exception:
            continue
    print(f"[DEBUG] Using column index: {col_idx} for {os.path.basename(filepath)}")
    return values


# ============================================================
# SECTION 3 — Motor: Chart & PDF Generation
# ============================================================

def add_motor_diagnostics(ax, months, month_values):
    """Overlay 12-month and 3-month median/std-dev status text on a motor chart."""
    medians = [safe_median(v) for v in month_values]
    stds    = [safe_stdev(v)  for v in month_values]
    dfm = pd.DataFrame({"Month": months, "Median": medians, "StdDev": stds}).dropna()
    dfm = dfm.sort_values("Month")

    def check(series, ok_text, bad_text):
        if len(series) < 2:
            return ok_text, "green"
        mu    = float(series.mean())
        sigma = float(series.std(ddof=1))
        if sigma == 0:
            return ok_text, "green"
        current = float(series.iloc[-1])
        ok = (mu - 3 * sigma) <= current <= (mu + 3 * sigma)
        return (ok_text, "green") if ok else (bad_text, "red")

    results = [
        check(dfm.tail(12)["Median"], "12 Month Median is OK",
              "12 Month Median is Bad"),
        check(dfm.tail(3)["Median"],  "3 Month Median is OK",
              "3 Month Median is Bad"),
        check(dfm.tail(12)["StdDev"], "12 Month Standard Deviation is Good",
              "12 Month Standard Deviation is Bad"),
        check(dfm.tail(3)["StdDev"],  "3 Month Standard Deviation is Good",
              "3 Month Standard Deviation is Bad"),
    ]
    for (txt, color), y in zip(results, [0.98, 0.92, 0.86, 0.80]):
        ax.text(0.01, y, txt, transform=ax.transAxes,
                fontsize=11, fontweight="bold", color=color, va="top")


def create_motor_pdf(output_path, months, month_values, labels=None):
    """Generate the motor trend analysis PDF (boxplot + std-dev chart + summary)."""
    pdf = FPDF()
    pdf.add_page()
    if os.path.exists(LOGO_FILE):
        pdf.image(LOGO_FILE, x=10, y=8, w=28)
        pdf.set_font("Arial", "B", 16)
        pdf.set_xy(45, 10)
        pdf.cell(0, 10, "Rexair Motor Test Trend Analysis", ln=True)

    month_labels = labels if labels else [m.strftime("%b %Y") for m in months]

    # Chart 1: Boxplot distribution
    plt.figure(figsize=(8.5, 4.2))
    plt.boxplot(month_values, tick_labels=month_labels, showfliers=False)
    plt.title("Input Power (W): Monthly Distribution")
    plt.ylabel("Watts")
    ax = plt.gca()
    add_motor_diagnostics(ax, months, month_values)
    plt.xticks(rotation=45)
    plt.tight_layout()
    img_box = "motor_boxplot_temp.png"
    plt.savefig(img_box, dpi=300)
    plt.close()
    pdf.image(img_box, x=10, y=30, w=190)

    # Chart 2: Standard deviation over time
    monthly_stds = [safe_stdev(v) for v in month_values]
    plt.figure(figsize=(8.5, 3.0))
    plt.plot(month_labels, monthly_stds, marker="o")
    plt.title("Standard Deviation Over Time")
    plt.ylabel("Std Dev (W)")
    plt.grid(True)
    ax = plt.gca()
    add_motor_diagnostics(ax, months, month_values)
    plt.xticks(rotation=45)
    plt.tight_layout()
    img_std = "motor_std_temp.png"
    plt.savefig(img_std, dpi=300)
    plt.close()
    pdf.image(img_std, x=10, y=125, w=190)

    # Page 2: Summary table
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Monthly Summary", ln=True)
    pdf.set_font("Arial", "", 11)
    for label, vals in zip(month_labels, month_values):
        mu  = safe_mean(vals)
        sd  = safe_stdev(vals)
        med = safe_median(vals)
        pdf.cell(0, 8,
                 f"{label}: Mean={mu:.2f} W, Median={med:.2f} W, StdDev={sd:.4f}",
                 ln=True)

    pdf.output(output_path)
    for tmp in (img_box, img_std):
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


# ============================================================
# SECTION 4 — Motor: Analysis Runner
# ============================================================

def run_motor_analysis(q=None, cancel_flag=None):
    """
    Collect motor Excel files, deduplicate, take the last 12 by date,
    label months with multiple files as 'Jul 1 / Jul 2', and generate PDF.
    """
    base       = os.getcwd()
    report_dir = os.path.join(base, REPORT_FOLDER_MOTOR)
    os.makedirs(report_dir, exist_ok=True)

    # Step 1 — collect all .xlsx / .xls files with a parseable date
    candidates = []
    for f in os.listdir(base):
        if f.lower().endswith((".xlsx", ".xls")):
            d = extract_date_from_filename(f)
            if d:
                candidates.append((d, f))

    # Step 2 — deduplicate: same stem + date → prefer .xlsx over .xls
    stem_groups = defaultdict(list)
    for d, f in candidates:
        stem = re.sub(r"\s*\(\d+\)$", "", os.path.splitext(f)[0])
        stem_groups[(d, stem)].append(f)

    deduped = []
    for (d, stem), fnames in stem_groups.items():
        xlsx   = [fn for fn in fnames if fn.lower().endswith(".xlsx")]
        chosen = xlsx[0] if xlsx else fnames[0]
        deduped.append((d, chosen))

    deduped.sort(key=lambda x: x[0])
    print(f"[MOTOR] Files after dedup: {len(deduped)}")
    for d, f in deduped:
        print(f"  {d.strftime('%Y-%m-%d')}  {f}")

    # Step 3 — last 12 files
    recent = deduped[-12:]

    # Step 4 — read power values
    file_entries = []
    for d, f in recent:
        if cancel_flag and cancel_flag.get("stop"):
            return
        vals = read_input_power(os.path.join(base, f))
        if vals:
            file_entries.append({"date": d, "filename": f, "values": vals})
        else:
            print(f"[WARN] No power values in: {f}")

    if not file_entries:
        print("No motor Input Power data found.")
        return

    # Step 5 — labels: months with >1 file get a numeric suffix
    final_counts = defaultdict(int)
    for e in file_entries:
        final_counts[e["date"].strftime("%Y-%m")] += 1

    month_seen = defaultdict(int)
    for e in file_entries:
        mk = e["date"].strftime("%Y-%m")
        month_seen[mk] += 1
        if final_counts[mk] > 1:
            e["label"] = f"{e['date'].strftime('%b %Y')} {month_seen[mk]}"
        else:
            e["label"] = e["date"].strftime("%b %Y")

    # Step 6 — generate PDF
    labels       = [e["label"]  for e in file_entries]
    dates        = [e["date"]   for e in file_entries]
    month_values = [e["values"] for e in file_entries]

    filename = f"Rexair_Motor_Test_Trend_Analysis_{datetime.now():%Y-%m}.pdf"
    outpath  = os.path.join(report_dir, filename)
    create_motor_pdf(outpath, dates, month_values, labels=labels)
    print(f"[INFO] Motor report created: {outpath}")
    open_file(outpath)


# ============================================================
# SECTION 5 — Magnet: OCR & Image Helpers
# ============================================================

def mag_render_pdf_page(pdf_file: str, page_num: int, dpi: int = 450) -> Image.Image:
    doc = fitz.open(pdf_file)
    try:
        page   = doc.load_page(page_num)
        zoom   = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix    = page.get_pixmap(matrix=matrix, alpha=False)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    finally:
        doc.close()


def mag_preprocess(img: Image.Image) -> Image.Image:
    """Threshold preprocessing — MedianFilter omitted to preserve decimal points."""
    gray = ImageOps.autocontrast(img.convert("L"))
    return gray.point(lambda x: 255 if x > 165 else 0)


def mag_preprocess_raw(img: Image.Image) -> Image.Image:
    """No-threshold variant for clean scans where thresholding merges glyphs."""
    return ImageOps.autocontrast(img.convert("L"))


def mag_ocr(img: Image.Image, psm: int = 6) -> str:
    return pytesseract.image_to_string(img, config=f"--oem 3 --psm {psm}")


def mag_get_page_text(doc: fitz.Document, pdf_file: str, page_num: int) -> str:
    """
    Try native PDF text; fall back to multi-rotation OCR.
    Scores candidates by keyword content so a rotated page's garbage
    output does not beat the correctly-oriented text.
    """
    native     = doc.load_page(page_num).get_text("text") or ""
    candidates = [native]

    if len(native.strip()) < 40:
        base_img = mag_render_pdf_page(pdf_file, page_num, dpi=450)
        for angle in (0, 90, 270):
            img  = base_img.rotate(angle, expand=True) if angle else base_img
            proc = mag_preprocess(img)
            for psm in (6, 11):
                text = mag_ocr(proc, psm=psm)
                if text:
                    candidates.append(text)

    def _score(s: str) -> int:
        keywords = [
            "date", "inspect", "customer", "order", "part", "drawing",
            "production", "sampling", "lot", "acuger", "rexair",
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec",
        ]
        base = len(re.sub(r"\s+", "", s))
        hits = sum(1 for kw in keywords if kw in s.lower())
        return base + hits * 200

    return max(candidates, key=_score)


# ============================================================
# SECTION 6 — Magnet: Text Normalization & Value Parsing
# ============================================================

def mag_normalize(text: str) -> str:
    if not text:
        return ""
    replacements = {
        "\u00b0": "", "\u00ba": "", "O": "0", "o": "0",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\uff1a": ":", "\uff0c": ",", "\u3002": ".", "\u00b7": ".", "\u2022": ".",
    }
    t = text
    for old, new in replacements.items():
        t = t.replace(old, new)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\r", "\n", t)
    return t


def parse_3x_values(text: str) -> List[float]:
    """
    Extract 3.xx measurements from OCR text.
    Pass 1 — explicit separator  (3.76, 3,76, 3:76, 3 76)
    Pass 2 — dropped decimal     (376 -> 3.76) with tighter floor to block noise.
    """
    t         = mag_normalize(text)
    sep_spans: set = set()
    values:    List[float] = []

    for m in re.finditer(r'(?<!\d)(3)[\.,:\s](\d{2})(?!\d)', t):
        val = float(f"3.{m.group(2)}")
        if 3.0 <= val <= 4.5:
            values.append(val)
            sep_spans.add(m.start())

    for m in re.finditer(r'(?<!\d)3(\d{2})(?!\d)', t):
        if any(abs(m.start() - s) <= 1 for s in sep_spans):
            continue
        val = float(f"3.{m.group(1)}")
        if 3.45 <= val <= MAG_USL:   # tighter floor blocks OCR noise
            values.append(val)

    return values


def dedupe_preserve_order(values: List[float], tol: float = 1e-9) -> List[float]:
    result = []
    for v in values:
        if not result or abs(result[-1] - v) > tol:
            result.append(v)
    return result


# ============================================================
# SECTION 7 — Magnet: Month & Measurement Extraction from PDF
# ============================================================

def mag_extract_month(text: str, filename: str) -> Optional[datetime]:
    t = mag_normalize(text)
    patterns = [
        r'([A-Za-z]{3,9})[\/\- ](\d{1,2})[\/\- ,]+(20\d{2})',
        r'(20\d{2})[\/\-_](\d{1,2})[\/\-_](\d{1,2})',
    ]
    m = re.search(patterns[0], t, flags=re.IGNORECASE)
    if m:
        try:
            month_num = datetime.strptime(m.group(1)[:3].title(), "%b").month
            return datetime(int(m.group(3)), month_num, 1)
        except ValueError:
            pass
    m = re.search(patterns[1], t)
    if m:
        return datetime(int(m.group(1)), int(m.group(2)), 1)
    for pat in (r'(20\d{2})[-_](\d{2})[-_](\d{2})', r'(20\d{2})(\d{2})(\d{2})'):
        m = re.search(pat, filename)
        if m:
            return datetime(int(m.group(1)), int(m.group(2)), 1)
    return None


def _extract_cells(col_img: Image.Image) -> List[float]:
    """
    Detect horizontal table lines and OCR each data cell individually (psm=7).
    Fallback for PDFs where whole-page OCR scrambles individual rows.
    """
    arr       = np.array(col_img.convert("L"))
    row_means = arr.mean(axis=1)
    is_dark   = row_means < 100

    lines: List[int] = []
    in_line = False
    start   = 0
    for i, d in enumerate(is_dark):
        if d and not in_line:
            in_line = True
            start = i
        elif not d and in_line:
            in_line = False
            lines.append((start + i) // 2)

    if len(lines) < 3:
        return []

    col_w  = col_img.width
    values: List[float] = []

    for i in range(1, len(lines) - 1):   # row 0 is the column header
        y1, y2 = lines[i] + 2, lines[i + 1] - 2
        if y2 - y1 < 5:
            continue
        cell     = col_img.crop((0, y1, col_w, y2))
        cell_big = cell.resize((cell.width * 4, cell.height * 4), Image.LANCZOS)

        for proc in (mag_preprocess(cell_big), mag_preprocess_raw(cell_big)):
            text = pytesseract.image_to_string(
                proc,
                config="--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789."
            ).strip()
            cell_vals = parse_3x_values(text)
            if cell_vals:
                values.extend(cell_vals)
                break

    return values


def extract_values_from_page2(doc: fitz.Document, pdf_file: str) -> List[float]:
    """
    Build multiple OCR extraction candidates from page 2 and return the best.
    Tries full-page, right-crop, targeted column, narrow band, bottom-half,
    and cell-by-cell extraction, both with and without thresholding.
    """
    candidates: List[List[float]] = []

    # 1) Native text layer
    native = doc.load_page(1).get_text("text") or ""
    native_vals = parse_3x_values(native)
    if native_vals:
        candidates.append(native_vals)

    # 2) Full-page OCR
    base_img  = mag_render_pdf_page(pdf_file, 1, dpi=500)
    full_proc = mag_preprocess(base_img)
    for psm in (4, 6, 11):
        vals = parse_3x_values(mag_ocr(full_proc, psm=psm))
        if vals:
            candidates.append(vals)

    w, h = base_img.size

    # 3) Right-side crop (0.45–0.99 width, full height)
    right_crop = base_img.crop((int(w * 0.45), int(h * 0.05), int(w * 0.99), h))
    for proc in (mag_preprocess(right_crop), mag_preprocess_raw(right_crop)):
        for psm in (4, 6, 11):
            vals = parse_3x_values(mag_ocr(proc, psm=psm))
            if vals:
                candidates.append(vals)

    # 4) Narrow far-right band (0.58–0.97 width)
    narrow_crop = base_img.crop((int(w * 0.58), int(h * 0.05), int(w * 0.97), h))
    for psm in (4, 6, 11):
        vals = parse_3x_values(mag_ocr(mag_preprocess(narrow_crop), psm=psm))
        if vals:
            candidates.append(vals)

    # 5) Targeted column crop (0.47–0.62 width) — confirmed data location
    target_crop = base_img.crop((int(w * 0.47), int(h * 0.06), int(w * 0.62), h))
    for proc in (mag_preprocess(target_crop), mag_preprocess_raw(target_crop)):
        for psm in (4, 6, 11):
            vals = parse_3x_values(mag_ocr(proc, psm=psm))
            if vals:
                candidates.append(vals)

    # 6) Bottom-half crop — catches rows 20-30 that top-biased passes miss
    bottom_crop = base_img.crop((int(w * 0.45), int(h * 0.55), int(w * 0.99), h))
    for psm in (4, 6, 11):
        vals = parse_3x_values(mag_ocr(mag_preprocess(bottom_crop), psm=psm))
        if vals:
            candidates.append(vals)

    # 7) Cell-by-cell fallback on the targeted column
    cell_vals = _extract_cells(target_crop)
    if cell_vals:
        candidates.append(cell_vals)

    # Select best candidate
    cleaned = [dedupe_preserve_order(c) for c in candidates if c]
    if not cleaned:
        return []

    exact = [c for c in cleaned if len(c) >= TARGET_VALUES]
    if exact:
        return min(exact, key=lambda c: len(c) - TARGET_VALUES)[:TARGET_VALUES]

    # If no candidate reaches TARGET_VALUES, try merging top + bottom
    if len(cleaned) >= 2:
        top    = max(cleaned, key=len)
        others = [c for c in cleaned if c is not top]
        if others:
            bottom = max(others, key=len)
            merged = top + [v for v in bottom if v not in top[-5:]]
            merged = dedupe_preserve_order(merged)
            if len(merged) > len(top):
                cleaned.append(merged)
                if len(merged) >= TARGET_VALUES:
                    return merged[:TARGET_VALUES]

    return max(cleaned, key=len)[:TARGET_VALUES]

# ============================================================
# SECTION 8 — Magnet: Cp/CpK & PDF Report
# ============================================================

def mag_compute_cp_cpk(mean: float, std: float) -> Tuple[float, float]:
    if std <= 0:
        return 0.0, 0.0
    cp  = (MAG_USL - MAG_LSL) / (6 * std)
    cpk = min((mean - MAG_LSL) / (3 * std), (MAG_USL - mean) / (3 * std))
    return cp, cpk


def create_magnet_pdf(output_path: str, rows: List[dict]) -> None:
    """Generate the magnet CpK trend report PDF with embedded status."""
    
    labels = [r["label"] for r in rows]
    cpks   = [r["cpk"]   for r in rows]

    # =========================
    # Create CpK Chart
    # =========================
    plt.figure(figsize=(8, 4))
    plt.plot(labels, cpks, marker="o")
    plt.axhline(CHART_TARGET_CPK, linestyle="--")
    plt.title("Magnet Timing CpK Trend")
    plt.ylabel("Cpk")
    plt.grid(True)

    # =========================
    # CpK Status (LATEST ONLY)
    # =========================
    latest_cpk = cpks[-1]

    if latest_cpk >= CHART_TARGET_CPK:
        status_text = "CpK is Acceptable"
        color = "green"
    else:
        status_text = "CpK is not Acceptable"
        color = "red"

    # Place text INSIDE chart
    plt.text(
        0.02, 0.95,
        status_text,
        transform=plt.gca().transAxes,
        fontsize=14,
        fontweight="bold",
        color=color,
        verticalalignment="top",
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
    )

    plt.xticks(rotation=45)
    plt.tight_layout()

    chart_file = "magnet_cpk_temp.png"
    plt.savefig(chart_file, dpi=200)
    plt.close()

    # =========================
    # Create PDF
    # =========================
    pdf = FPDF()
    pdf.add_page()

    if os.path.exists(LOGO_FILE):
        pdf.image(LOGO_FILE, x=10, y=8, w=28)

    pdf.set_font("Arial", "B", 16)
    pdf.set_xy(45, 10)
    pdf.cell(0, 10, "Rexair Magnet Timing CpK Report", ln=True)

    # Insert chart ONLY (status already inside image)
    pdf.image(chart_file, x=10, y=30, w=190)

    # =========================
    # Data Table
    # =========================
    pdf.set_y(120)
    pdf.set_font("Arial", "", 11)

    for row in rows:
        pdf.cell(
            0, 8,
            f"{row['label']}   "
            f"Mean:{row['mean']:.4f}   "
            f"Std:{row['std']:.4f}   "
            f"Cp:{row['cp']:.2f}   "
            f"Cpk:{row['cpk']:.2f}",
            ln=True
        )

    pdf.output(output_path)

    # Cleanup temp image
    try:
        if os.path.exists(chart_file):
            os.remove(chart_file)
    except Exception:
        pass


# ============================================================
# SECTION 9 — Magnet: PDF Processing & Analysis Runner
# ============================================================

def process_magnet_pdf(pdf_file: str) -> Optional[dict]:
    """
    Open a magnet inspection PDF, extract the date from page 1 and the
    30 angle measurements from page 2, compute stats, return a result dict.
    """
    print(f"\nProcessing: {pdf_file}")
    try:
        doc = fitz.open(pdf_file)
    except Exception as exc:
        print(f"  Skipped — could not open: {exc}")
        return None

    try:
        if doc.page_count < 2:
            print(f"  Skipped — only {doc.page_count} page(s)")
            return None

        page1_text = mag_get_page_text(doc, pdf_file, 0)
        month      = mag_extract_month(page1_text, pdf_file)
        if not month:
            print("  Skipped — month not detected")
            return None

        values = extract_values_from_page2(doc, pdf_file)
        if len(values) < MIN_VALUES:
            print(f"  Skipped — insufficient values: {len(values)}")
            return None

        arr  = np.array(values, dtype=float)
        mean = float(arr.mean())
        std  = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        cp, cpk = mag_compute_cp_cpk(mean, std)

        print(f"  OK  n={len(values)}  mean={mean:.4f}  std={std:.4f}  cpk={cpk:.2f}")
        return {
            "file":     pdf_file,
            "month_dt": month,
            "values":   values,
            "mean":     mean,
            "std":      std,
            "cp":       cp,
            "cpk":      cpk,
        }
    finally:
        doc.close()


def run_magnet_analysis(q=None, cancel_flag=None):
    """
    Scan the working directory for magnet inspection PDFs, extract angle data,
    compute Cp/CpK, and write a trend report to the Magnet Reports folder.
    Skips any PDF whose name starts with 'Rexair_Magnet_Timing_CpK_Report_'.
    """
    base       = os.getcwd()
    report_dir = os.path.join(base, REPORT_FOLDER_MAGNET)
    os.makedirs(report_dir, exist_ok=True)

    pdf_files = sorted(
        f for f in os.listdir(base)
        if f.lower().endswith(".pdf")
        and not f.startswith("Rexair_Magnet_Timing_CpK_Report_")
    )

    if not pdf_files:
        print("No magnet PDF files found.")
        return

    rows: List[dict] = []
    for fname in pdf_files:
        if cancel_flag and cancel_flag.get("stop"):
            return
        row = process_magnet_pdf(os.path.join(base, fname))
        if row:
            rows.append(row)

    if not rows:
        print("\nNo valid magnet datasets found. Report not created.")
        return

    # Sort by date then assign labels; multi-file months get "Jan 1", "Jan 2", etc.
    rows.sort(key=lambda r: (r["month_dt"], r["file"]))

    month_counts = defaultdict(int)
    for r in rows:
        month_counts[r["month_dt"]] += 1

    month_seen = defaultdict(int)
    for r in rows:
        dt = r["month_dt"]
        month_seen[dt] += 1
        if month_counts[dt] > 1:
            r["label"] = f"{dt.strftime('%b %Y')} {month_seen[dt]}"
        else:
            r["label"] = dt.strftime("%b %Y")

    outpath = os.path.join(
        report_dir,
        f"Rexair_Magnet_Timing_CpK_Report_{datetime.now():%Y-%m}.pdf"
    )
    create_magnet_pdf(outpath, rows)
    print(f"\n[INFO] Magnet report created: {outpath}")
    open_file(outpath)


# ============================================================
# SECTION 10 — GUI & Entry Point
# ============================================================

def start_threaded(funcs):
    """Run analysis functions on a background thread with a progress window."""
    root = tk.Tk()
    root.title("Running...")
    root.geometry("420x120")
    tk.Label(root, text="Processing...", font=("Segoe UI", 12)).pack(pady=10)
    bar = ttk.Progressbar(root, mode="indeterminate")
    bar.pack(pady=6, fill="x", padx=20)
    bar.start()

    q           = queue.Queue()
    cancel_flag = {"stop": False}

    def worker():
        for func in funcs:
            try:
                func(q, cancel_flag)
            except Exception as e:
                print(f"[ERROR] {e}")
        q.put("done")

    def check_done():
        try:
            if q.get_nowait() == "done":
                bar.stop()
                root.destroy()
                return
        except queue.Empty:
            pass
        root.after(200, check_done)

    def on_close():
        cancel_flag["stop"] = True
        bar.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    threading.Thread(target=worker, daemon=True).start()
    root.after(200, check_done)
    root.mainloop()


def launch_gui():
    root = tk.Tk()
    root.title("Motor & Magnet Analysis")
    root.geometry("520x300")

    tk.Label(root, text="Motor & Magnet Analysis",
             font=("Segoe UI", 14, "bold")).pack(pady=12)

    tk.Button(root, text="Run Motor Analysis", width=45,
              command=lambda: start_threaded([run_motor_analysis])).pack(pady=8)

    tk.Button(root, text="Run Magnet Analysis", width=45,
              command=lambda: start_threaded([run_magnet_analysis])).pack(pady=8)

    tk.Button(root, text="Run Both", width=45,
              command=lambda: start_threaded([run_motor_analysis,
                                              run_magnet_analysis])).pack(pady=8)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
