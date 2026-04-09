# ============================================================
# SECTION 1
# Motor & Magnet Analysis - Unified Script (UPDATED)
# ============================================================

import os
import re
import sys
import platform
import subprocess
from datetime import datetime
import threading
import queue
from collections import defaultdict

import numpy as np
import pandas as pd

# Use a non-GUI backend for Matplotlib (safe for threading)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fpdf import FPDF  # PDF generation
import tkinter as tk   # GUI
from tkinter import ttk
import fitz            # PyMuPDF for PDF reading (and OCR)

# Optional OCR dependencies for scanned PDFs (pytesseract)
try:
    from PIL import Image, ImageEnhance
    import pytesseract
    _HAS_TESSERACT = True
except Exception:
    Image = None
    ImageEnhance = None
    pytesseract = None
    _HAS_TESSERACT = False

# ============================================================
# Configuration
# ============================================================

REPORT_FOLDER_MOTOR = "Motor Reports"
REPORT_FOLDER_MAGNET = "Magnet Reports"
LOGO_FILE = "rexair_logo.png"  # optional company logo for PDF

# Spec target and tolerance for magnet timing (for Cp/CpK calculation)
MAGNET_TARGET = 3.7
MAGNET_TOL = 0.5
LSL = MAGNET_TARGET - MAGNET_TOL  # 3.2
USL = MAGNET_TARGET + MAGNET_TOL  # 4.2

os.makedirs(REPORT_FOLDER_MOTOR, exist_ok=True)
os.makedirs(REPORT_FOLDER_MAGNET, exist_ok=True)

# ============================================================
# Utility: Open File
# ============================================================

def open_file(filepath):
    """Open a file using the default application, depending on OS."""
    try:
        if platform.system() == "Windows":
            os.startfile(filepath)
        elif platform.system() == "Darwin":
            subprocess.run(["open", filepath], check=False)
        else:
            subprocess.run(["xdg-open", filepath], check=False)
    except Exception as e:
        print(f"[WARN] Could not open file: {e}")

# ============================================================
# Numeric Safety Helpers
# ============================================================

def safe_numeric_array(values):
    """Convert a list of values to a NumPy array of floats, ignoring invalid entries."""
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
    if arr.size < 2:
        return 0.0
    return float(np.std(arr, ddof=1))

# ============================================================
# SECTION 2
# Motor Data Helpers
# ============================================================

def extract_date_from_filename(fname):
    """
    Extract date from filename of form 'C6521YYMMDD'.
    Example: 'C6521240517' -> datetime(2024, 05, 17)
    """
    m = re.search(r"C6521(\d{6})", fname)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%y%m%d")
    except Exception:
        return None

def find_input_power_column(df):
    """Find the column index in DataFrame `df` that contains motor Input Power data."""
    for i in range(min(10, len(df))):
        row = df.iloc[i].astype(str).tolist()
        for j, cell in enumerate(row):
            cell_lower = cell.lower()
            if "high speed(open)" in cell_lower and "input power" in cell_lower:
                return j

    for i in range(min(10, len(df))):
        row = df.iloc[i].astype(str).tolist()
        for j, cell in enumerate(row):
            if "input power" in cell.lower():
                return j

    return None

def read_input_power(filepath):
    """Read motor input power values from the Excel file."""
    try:
        df = pd.read_excel(filepath, header=None, engine="openpyxl")
    except Exception as e:
        print(f"[ERROR] Could not read excel: {filepath} -> {e}")
        return []

    col_idx = find_input_power_column(df)
    if col_idx is None:
        print(f"[WARN] Could not find Input Power column in {os.path.basename(filepath)}")
        return []

    values = []
    for v in df.iloc[:, col_idx].tolist():
        try:
            fv = float(v)
            if np.isfinite(fv):
                values.append(fv)
        except Exception:
            continue
    return values

# ============================================================
# Motor Diagnostics
# ============================================================

def add_motor_diagnostics(ax, months, month_values):
    """
    Add diagnostic text to the motor charts indicating if the last 12/3 months
    medians and standard deviations are within control (3σ limits).
    """
    medians = [safe_median(v) for v in month_values]
    stds = [safe_stdev(v) for v in month_values]
    dfm = pd.DataFrame({"Month": months, "Median": medians, "StdDev": stds}).dropna()
    dfm = dfm.sort_values("Month")

    def check(series, ok_text, bad_text):
        if len(series) < 2:
            return ok_text, "green"
        mu = float(series.mean())
        sigma = float(series.std(ddof=1))
        if sigma == 0:
            return ok_text, "green"
        lower = mu - 3.0 * sigma
        upper = mu + 3.0 * sigma
        current_val = float(series.iloc[-1])
        return (ok_text, "green") if (lower <= current_val <= upper) else (bad_text, "red")

    last_12 = dfm.tail(12)
    last_3 = dfm.tail(3)
    results = [
        check(last_12["Median"], "12 Month Median is OK", "12 Month Median is Bad"),
        check(last_3["Median"], "3 Month Median is OK", "3 Month Median is Bad"),
        check(last_12["StdDev"], "12 Month Standard Deviation is Good", "12 Month Standard Deviation is Bad"),
        check(last_3["StdDev"], "3 Month Standard Deviation is Good", "3 Month Standard Deviation is Bad"),
    ]

    y_positions = [0.98, 0.92, 0.86, 0.80]
    for (txt, color), y in zip(results, y_positions):
        ax.text(0.01, y, txt, transform=ax.transAxes,
                fontsize=11, fontweight="bold", color=color, va="top")

# ============================================================
# PDF Report: Motor Data
# ============================================================

def create_motor_pdf(output_path, months, month_values):
    """Generate a PDF report for motor input power trends."""
    pdf = FPDF()
    pdf.add_page()

    if os.path.exists(LOGO_FILE):
        pdf.image(LOGO_FILE, x=10, y=8, w=28)

    pdf.set_font("Arial", "B", 16)
    pdf.set_xy(45, 10)
    pdf.cell(0, 10, "Rexair Motor Test Trend Analysis", ln=True)

    # Chart 1: Distribution per month (boxplot)
    plt.figure(figsize=(8.5, 4.2))
    month_labels = [m.strftime("%b %Y") for m in months]
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

    # Page 2: Summary
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Monthly Summary", ln=True)
    pdf.set_font("Arial", "", 11)

    for m, vals in zip(months, month_values):
        mu = safe_mean(vals)
        sd = safe_stdev(vals)
        med = safe_median(vals)
        pdf.cell(0, 8, f"{m.strftime('%Y-%m')}: Mean={mu:.2f} W, Median={med:.2f} W, StdDev={sd:.4f}", ln=True)

    pdf.output(output_path)

    for tmp in (img_box, img_std):
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

# ============================================================
# SECTION 3
# Magnet Data Analysis Helpers
# ============================================================

def compute_cp_cpk(mean, std):
    """Compute Cp and CpK from mean/std and fixed spec limits."""
    if std is None or std <= 0:
        return 0.0, 0.0
    cp = (USL - LSL) / (6.0 * std)
    cpk = min((mean - LSL) / (3.0 * std), (USL - mean) / (3.0 * std))
    return cp, cpk

def dump_magnet_debug(base_dir, pdf_filename, extracted_text):
    """Write extracted text to a debug file for troubleshooting."""
    debug_dir = os.path.join(base_dir, "Magnet Debug")
    os.makedirs(debug_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(pdf_filename))[0]
    out_txt = os.path.join(debug_dir, f"{base}_textdump.txt")
    try:
        with open(out_txt, "w", encoding="utf-8", errors="ignore") as f:
            f.write(extracted_text)
        print(f"[MAGNET] Debug text dumped to: {out_txt}")
    except Exception as e:
        print(f"[MAGNET] Could not write debug dump: {e}")

def extract_pages_text(pdf_path, ocr_language="eng"):
    """
    Return a list of page texts. Tries standard text extraction first.
    If page text is very sparse and OCR is available, OCR that page.
    """
    page_texts = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return [], f"read_error:{e}"

    try:
        for page in doc:
            txt = page.get_text("text")
            txt = txt if txt else ""

            # OCR fallback for sparse pages
            if len(txt.strip()) < 30 and _HAS_TESSERACT:
                try:
                    tp = page.get_textpage_ocr(language=ocr_language, dpi=300, full=True)
                    txt_ocr = page.get_text("text", textpage=tp)
                    if len(txt_ocr.strip()) > len(txt.strip()):
                        txt = txt_ocr
                except Exception:
                    pass

            page_texts.append(txt)
    finally:
        doc.close()

    return page_texts, "ok"

def extract_inspection_month_from_text(page1_text):
    """
    Extract inspection date from page 1 text.
    Expected examples:
      INSPECTION : Apr/14/2025
      INSPECTION:Sep/25/2025
    Returns a datetime with day=1, or None.
    """
    if not page1_text:
        return None

    patterns = [
        r"INSPECTION\s*[:：]?\s*([A-Za-z]{3}/\d{1,2}/\d{4})",
        r"Inspection\s*[:：]?\s*([A-Za-z]{3}/\d{1,2}/\d{4})",
        r"INSPECTION DATE\s*[:：]?\s*([A-Za-z]{3}/\d{1,2}/\d{4})",
    ]

    for pat in patterns:
        m = re.search(pat, page1_text, flags=re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            try:
                dt = datetime.strptime(raw, "%b/%d/%Y")
                return dt.replace(day=1)
            except Exception:
                pass

    return None

def normalize_for_numeric_ocr(text):
    """Normalize OCR and punctuation quirks for numeric extraction."""
    if not text:
        return ""
    t = text.replace("\u00a0", " ")
    t = t.replace("O", "0").replace("o", "0")
    t = t.replace("I", "1").replace("l", "1")
    t = t.replace("S", "5")
    t = t.replace("，", ",").replace("：", ":").replace("；", ";")
    t = t.replace("·", ".").replace(",", ".")
    t = t.replace("º", "°")
    return t

def parse_37_values_from_page2_text(page2_text, target_count=30):
    """
    Extract only the 3.7 ± 0.5° right-hand column values from page 2 text.
    Strategy:
      1) Find lines that look like row data with row index + 15° value + 3.x value
      2) Extract the right-most 3.x value
      3) Require roughly 30 values
    """
    if not page2_text:
        return []

    t = normalize_for_numeric_ocr(page2_text)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    values = []

    for line in lines:
        # Look for row number at the front and a 3.xx value later in the line
        # Example:
        # 1 15°16´ 3.80°
        # 2 15°24´ 3.82°
        # OCR may drop degree symbols, so keep it flexible
        row_match = re.match(r"^\s*(\d{1,2})\b", line)
        if not row_match:
            continue

        nums = re.findall(r"\b([34]\.\d{1,2})\b", line)
        if nums:
            try:
                v = float(nums[-1])  # right-most decimal on the row
                if 3.0 <= v <= 4.5:
                    values.append(v)
            except Exception:
                continue

    # If that worked, return the first 30 row values
    if len(values) >= 20:
        return values[:target_count]

    # Fallback: scan only after the 3.7 header
    header_match = re.search(r"3\s*\.?\s*7\s*(?:±|\+/?-)\s*0\s*\.?\s*5", t)
    scan_text = t[header_match.start():] if header_match else t

    vals = []
    for m in re.finditer(r"\b([34]\.\d{1,2})\b", scan_text):
        try:
            v = float(m.group(1))
        except Exception:
            continue
        if 3.0 <= v <= 4.5:
            vals.append(v)
        if len(vals) >= target_count:
            return vals[:target_count]

    # OCR sometimes drops decimal point, e.g., 380 -> 3.80
    for m in re.finditer(r"\b([34]\d{2})\b", scan_text):
        s = m.group(1)
        try:
            v = float(s) / 100.0
        except Exception:
            continue
        if 3.0 <= v <= 4.5:
            vals.append(v)
        if len(vals) >= target_count:
            return vals[:target_count]

    return vals[:target_count]

def extract_timing_values_from_pdf_page_images(pdf_path, target_page_index=1, dpi=450):
    """
    OCR-based fallback for page 2 only. Crops the right side of the page and
    attempts to extract the 3.7° column values.
    """
    if not _HAS_TESSERACT:
        return []

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return []

    try:
        if doc.page_count <= target_page_index:
            return []

        page = doc.load_page(target_page_index)
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples).convert("L")
        w, h = img.size

        # Crop likely right-column area on page 2
        crop_boxes = [
            (0.52, 0.08, 0.96, 0.96),
            (0.58, 0.10, 0.98, 0.96),
            (0.50, 0.12, 0.95, 0.98),
        ]

        configs = [
            '--psm 6 -c tessedit_char_whitelist=0123456789.°/-±',
            '--psm 4 -c tessedit_char_whitelist=0123456789.°/-±',
            '--psm 11 -c tessedit_char_whitelist=0123456789.°/-±',
        ]

        for (lx, ty, rx, by) in crop_boxes:
            crop = img.crop((int(w * lx), int(h * ty), int(w * rx), int(h * by)))
            crop = ImageEnhance.Contrast(crop).enhance(3.0)
            crop = crop.point(lambda p: 255 if p > 180 else 0)

            for cfg in configs:
                try:
                    ocr_text = pytesseract.image_to_string(crop, config=cfg)
                except Exception:
                    continue

                vals = parse_37_values_from_page2_text(ocr_text, target_count=30)
                if len(vals) >= 20:
                    return vals[:30]

        return []
    finally:
        doc.close()

def read_magnet_timing_from_excel(filepath):
    """
    Read 3.7 ± 0.5° column from Excel, if magnet timing data also exists in xlsx.
    """
    try:
        df = pd.read_excel(filepath, sheet_name=1, header=None, engine="openpyxl")
    except Exception as e:
        print(f"[MAGNET] Could not read excel: {filepath} -> {e}")
        return []

    header_pat = re.compile(r"3\.7\s*(?:±|\+/-)\s*0\.5\s*°?", re.IGNORECASE)
    col_idx = None

    for i in range(min(15, len(df))):
        row = df.iloc[i].astype(str).tolist()
        for j, cell in enumerate(row):
            if header_pat.search(cell):
                col_idx = j
                break
        if col_idx is not None:
            break

    if col_idx is None:
        non_empty_cols = [c for c in range(df.shape[1]) if df.iloc[:, c].notna().sum() > 0]
        col_idx = non_empty_cols[-1] if non_empty_cols else None

    if col_idx is None:
        print(f"[MAGNET] Could not locate '3.7 +/- 0.5' column in {os.path.basename(filepath)}")
        return []

    values = []
    for v in df.iloc[:, col_idx].tolist():
        try:
            s = str(v).replace("°", "").strip()
            if s == "" or s.lower() in {"nan", "none"}:
                continue
            fv = float(s)
            if 3.0 <= fv <= 4.5:
                values.append(fv)
        except Exception:
            continue

    return values[:30]

def make_duplicate_key(values):
    """Round for stable comparison across tiny OCR/float formatting noise."""
    return tuple(round(float(v), 4) for v in values)

def fit_note_text(pdf, text, width=190):
    """
    Write wrapped text in red to the PDF.
    """
    pdf.set_text_color(255, 0, 0)
    pdf.set_font("Arial", "B", 10)
    pdf.multi_cell(width, 6, text)
    pdf.set_text_color(0, 0, 0)

# ============================================================
# PDF Report: Magnet Data
# ============================================================

def create_magnet_pdf(output_path, rows, incomplete_files=None, duplicate_log=None):
    """
    Create magnet report in the style you showed:
    - title
    - CpK chart
    - Per-File Summary Table
    - red notes at bottom for incomplete / duplicate reports
    """
    incomplete_files = incomplete_files or []
    duplicate_log = duplicate_log or []

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    if os.path.exists(LOGO_FILE):
        pdf.image(LOGO_FILE, x=10, y=8, w=28)

    pdf.set_font("Arial", "B", 16)
    pdf.set_xy(45, 10)
    pdf.cell(0, 10, "Rexair Magnet Timing CpK Report", ln=True)

    labels = [r.get("label") or r["month_dt"].strftime("%b %Y") for r in rows]
    cpks = [r["cpk"] for r in rows]

    # Chart
    plt.figure(figsize=(8.5, 4.0))
    plt.plot(labels, cpks, marker="o")
    plt.axhline(1.33, color="black", linestyle="--", linewidth=1.3)
    plt.title("Magnet Timing CpK Over Time (Per File)")
    plt.xlabel("File within Month")
    plt.ylabel("CpK")
    ymax = max(4.0, (max(cpks) + 0.25) if cpks else 4.0)
    plt.ylim(0, ymax)
    plt.grid(True)

    ax = plt.gca()
    latest_cpk = cpks[-1] if cpks else None
    if latest_cpk is not None:
        if latest_cpk >= 1.33:
            ax.text(0.5, 0.5, "CpK is Good", transform=ax.transAxes,
                    fontsize=18, fontweight="bold", color="green",
                    ha="center", va="center")
        else:
            ax.text(0.5, 0.5, "CpK is BAD", transform=ax.transAxes,
                    fontsize=18, fontweight="bold", color="red",
                    ha="center", va="center")

    plt.xticks(rotation=45)
    plt.tight_layout()
    img = "magnet_cpk_temp.png"
    plt.savefig(img, dpi=300)
    plt.close()

    pdf.image(img, x=10, y=30, w=190)

    # Summary section
    pdf.set_xy(10, 125)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Per-File Summary Table", ln=True)
    pdf.set_font("Arial", "", 11)

    for r in rows:
        label = r.get("label") or r["month_dt"].strftime("%B %Y")
        line = (
            f"{label} "
            f"Mean: {r['mean']:.4f}  "
            f"Std: {r['std']:.4f}  "
            f"Cp: {r['cp']:.2f}  "
            f"CpK: {r['cpk']:.2f}"
        )
        pdf.cell(0, 8, line, ln=True)

    # Bottom notes in red
    if incomplete_files or duplicate_log:
        pdf.ln(4)

        if incomplete_files:
            fit_note_text(pdf, "INCOMPLETE REPORT RECEIVED")
            pdf.set_font("Arial", "", 10)
            pdf.set_text_color(255, 0, 0)
            for f in incomplete_files:
                pdf.multi_cell(190, 5, f" - {f}")
            pdf.set_text_color(0, 0, 0)

        if duplicate_log:
            pdf.ln(2)
            fit_note_text(pdf, "DUPLICATE REPORTS INCLUDED")
            pdf.set_font("Arial", "", 10)
            pdf.set_text_color(255, 0, 0)
            for dup_file, kept_file in duplicate_log:
                pdf.multi_cell(190, 5, f" - {dup_file} duplicates {kept_file}")
            pdf.set_text_color(0, 0, 0)

    pdf.output(output_path)

    try:
        if os.path.exists(img):
            os.remove(img)
    except Exception:
        pass

# ============================================================
# SECTION 4
# Analysis Runner: Motor Data
# ============================================================

def run_motor_analysis(q=None, cancel_flag=None):
    """Process all motor Excel files and generate trend analysis PDF."""
    base = os.getcwd()
    report_dir = os.path.join(base, REPORT_FOLDER_MOTOR)
    os.makedirs(report_dir, exist_ok=True)

    files = [f for f in os.listdir(base) if f.lower().endswith(".xlsx")]
    dated = []

    for f in files:
        d = extract_date_from_filename(f)
        if d:
            dated.append((d, f))

    dated.sort(key=lambda x: x[0])
    monthly_data = {}

    for d, f in dated:
        if cancel_flag and cancel_flag.get("stop"):
            return
        month_key = d.strftime("%Y-%m")
        vals = read_input_power(os.path.join(base, f))
        if vals:
            monthly_data.setdefault(month_key, []).extend(vals)

    if not monthly_data:
        print("❌ No motor Input Power data found.")
        return

    all_months = sorted(datetime.strptime(m, "%Y-%m") for m in monthly_data.keys())
    recent_months = all_months[-12:]
    month_values = [monthly_data[m.strftime("%Y-%m")] for m in recent_months]

    filename = f"Rexair_Motor_Test_Trend_Analysis_{datetime.now():%Y-%m}.pdf"
    outpath = os.path.join(report_dir, filename)
    create_motor_pdf(outpath, recent_months, month_values)
    print(f"[INFO] ✅ Motor report created: {outpath}")
    open_file(outpath)

# ============================================================
# Analysis Runner: Magnet Data
# ============================================================

def run_magnet_analysis(q=None, cancel_flag=None):
    """
    Process all magnet timing PDF and Excel files, using:
    - page 1 inspection date as month
    - page 2 right-hand 3.7° column values
    - duplicate detection by same month + identical values
    - incomplete report logging
    """
    base = os.getcwd()
    print(f"[DEBUG] Current working directory: {base}")
    print(f"[DEBUG] Files in directory: {os.listdir(base)}")

    report_dir = os.path.join(base, REPORT_FOLDER_MAGNET)
    os.makedirs(report_dir, exist_ok=True)

    rows = []
    incomplete_files = []
    duplicate_log = []

    # Registry by month to compare duplicates only within same inspection month
    month_value_registry = defaultdict(dict)  # {month_key: {dup_key: source_file}}

    # ----------------------------
    # Process PDFs
    # ----------------------------
    pdf_files = [f for f in os.listdir(base) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("[MAGNET] No PDF files found in script folder for magnet analysis.")

    for f in sorted(pdf_files):
        if cancel_flag and cancel_flag.get("stop"):
            return

        full_path = os.path.join(base, f)
        page_texts, mode = extract_pages_text(full_path, ocr_language="eng")

        if mode.startswith("read_error") or not page_texts:
            print(f"[MAGNET] SKIP (can't read): {f} -> {mode}")
            incomplete_files.append(f)
            continue

        if len(page_texts) < 2:
            print(f"[MAGNET] INCOMPLETE REPORT RECEIVED (missing second page): {f}")
            incomplete_files.append(f)
            continue

        page1_text = page_texts[0]
        page2_text = page_texts[1]

        month_dt = extract_inspection_month_from_text(page1_text)
        if month_dt is None:
            print(f"[MAGNET] INCOMPLETE REPORT RECEIVED (inspection date not found): {f}")
            incomplete_files.append(f)
            dump_magnet_debug(base, f, "\n\n---PAGE 1---\n" + page1_text + "\n\n---PAGE 2---\n" + page2_text)
            continue

        raw_vals = parse_37_values_from_page2_text(page2_text, target_count=30)

        # OCR image fallback if page text did not yield enough values
        if len(raw_vals) < 20:
            img_vals = extract_timing_values_from_pdf_page_images(full_path, target_page_index=1)
            if len(img_vals) >= 20:
                raw_vals = img_vals
                print(f"[MAGNET] Image OCR used for page 2 timing values: {f}")

        if len(raw_vals) < 20:
            print(f"[MAGNET] INCOMPLETE REPORT RECEIVED (page 2 timing values not found): {f}")
            incomplete_files.append(f)
            dump_magnet_debug(base, f, "\n\n---PAGE 1---\n" + page1_text + "\n\n---PAGE 2---\n" + page2_text)
            continue

        raw_vals = raw_vals[:30]
        dup_key = make_duplicate_key(raw_vals)
        month_key = month_dt.strftime("%Y-%m")

        # Duplicate detection only within same month
        if dup_key in month_value_registry[month_key]:
            kept_file = month_value_registry[month_key][dup_key]
            duplicate_log.append((f, kept_file))
            print(f"[MAGNET] DUPLICATE detected: {f} duplicates {kept_file}")
            continue

        month_value_registry[month_key][dup_key] = f

        arr = np.array(raw_vals, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        cp, cpk = compute_cp_cpk(mean, std)

        rows.append({
            "month_dt": month_dt,
            "mean": mean,
            "std": std,
            "cp": cp,
            "cpk": cpk,
            "source_file": f,
            "raw_values": raw_vals,
        })

        print(
            f"[MAGNET] OK(PDF): {f} "
            f"{month_dt.strftime('%b %Y')} "
            f"n={len(raw_vals)} mean={mean:.4f} std={std:.4f} cp={cp:.2f} cpk={cpk:.2f}"
        )

    # ----------------------------
    # Process Excel files (optional)
    # Keeps existing functionality if magnet xlsx files are present
    # ----------------------------
    xlsx_files = [f for f in os.listdir(base) if f.lower().endswith(".xlsx") and 'magnet' in f.lower()]
    for f in sorted(xlsx_files):
        if cancel_flag and cancel_flag.get("stop"):
            return

        full = os.path.join(base, f)
        vals30 = read_magnet_timing_from_excel(full)
        if len(vals30) < 20:
            continue

        arr = np.array(vals30[:30], dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        cp, cpk = compute_cp_cpk(mean, std)

        # For Excel fallback, use file modified date month
        month_dt = datetime.fromtimestamp(os.path.getmtime(full)).replace(day=1)
        month_key = month_dt.strftime("%Y-%m")
        dup_key = make_duplicate_key(arr.tolist())

        if dup_key in month_value_registry[month_key]:
            kept_file = month_value_registry[month_key][dup_key]
            duplicate_log.append((f, kept_file))
            print(f"[MAGNET] DUPLICATE detected: {f} duplicates {kept_file}")
            continue

        month_value_registry[month_key][dup_key] = f

        rows.append({
            "month_dt": month_dt,
            "mean": mean,
            "std": std,
            "cp": cp,
            "cpk": cpk,
            "source_file": f,
            "raw_values": arr.tolist(),
        })

        print(
            f"[MAGNET] OK(Excel): {f} "
            f"{month_dt.strftime('%b %Y')} "
            f"n={len(arr)} mean={mean:.4f} std={std:.4f} cp={cp:.2f} cpk={cpk:.2f}"
        )

    if not rows:
        print("❌ No magnet data found. Check Magnet Debug folder and report structure.")
        return

    # Sort by month then file name
    rows.sort(key=lambda r: (r["month_dt"], r["source_file"].lower()))

    # Label each entry per month, e.g. "April 1", "April 2"
    month_counters = defaultdict(int)
    for r in rows:
        month_counters[r["month_dt"]] += 1
        r["label"] = f"{r['month_dt'].strftime('%B')} {month_counters[r['month_dt']]}"

    filename = f"Rexair_Magnet_Timing_CpK_Report_{datetime.now():%Y-%m}.pdf"
    outpath = os.path.join(report_dir, filename)

    create_magnet_pdf(
        outpath,
        rows,
        incomplete_files=incomplete_files,
        duplicate_log=duplicate_log
    )

    print(f"[INFO] ✅ Magnet report created: {outpath}")

    if incomplete_files:
        print("\n[MAGNET] INCOMPLETE REPORT RECEIVED:")
        for f in incomplete_files:
            print(f"  - {f}")

    if duplicate_log:
        print("\n[MAGNET] DUPLICATE REPORTS INCLUDED:")
        for dup_file, kept_file in duplicate_log:
            print(f"  - {dup_file} duplicates {kept_file}")

    open_file(outpath)

# ============================================================
# SECTION 5
# GUI / Threading
# ============================================================

def start_threaded(funcs):
    """Run a list of functions in sequence on a background thread."""
    root = tk.Tk()
    root.title("Running...")
    root.geometry("420x120")

    tk.Label(root, text="Processing...", font=("Segoe UI", 12)).pack(pady=10)
    bar = ttk.Progressbar(root, mode="indeterminate")
    bar.pack(pady=6, fill="x", padx=20)
    bar.start()

    q = queue.Queue()
    cancel_flag = {"stop": False}

    def worker():
        for func in funcs:
            try:
                func(q, cancel_flag)
            except Exception as e:
                print(f"[ERROR] Worker exception: {e}")
        q.put("done")

    def check_done():
        try:
            msg = q.get_nowait()
            if msg == "done":
                try:
                    bar.stop()
                except Exception:
                    pass
                try:
                    root.destroy()
                except Exception:
                    pass
                return
        except queue.Empty:
            pass
        root.after(200, check_done)

    def on_close():
        cancel_flag["stop"] = True
        try:
            bar.stop()
        except Exception:
            pass
        try:
            root.destroy()
        except Exception:
            pass

    root.protocol("WM_DELETE_WINDOW", on_close)
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    root.after(200, check_done)
    root.mainloop()

def launch_gui():
    """Launch the Tkinter GUI for interactive use."""
    root = tk.Tk()
    root.title("Motor & Magnet Analysis")
    root.geometry("520x300")

    tk.Label(root, text="Motor & Magnet Analysis", font=("Segoe UI", 14, "bold")).pack(pady=12)

    tk.Button(
        root,
        text="Run Motor Analysis",
        width=45,
        command=lambda: start_threaded([run_motor_analysis])
    ).pack(pady=8)

    tk.Button(
        root,
        text="Run Magnet Analysis (PDF + Excel)",
        width=45,
        command=lambda: start_threaded([run_magnet_analysis])
    ).pack(pady=8)

    tk.Button(
        root,
        text="Run Both",
        width=45,
        command=lambda: start_threaded([run_motor_analysis, run_magnet_analysis])
    ).pack(pady=8)

    root.mainloop()

# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    launch_gui()