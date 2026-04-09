# ============================================================
# Part 1
# Motor & Magnet Analysis Agent
# Multi-files per month labeled "January 1", "January 2", etc.
# ============================================================

import os
import re
import sys
import platform
import subprocess
from datetime import datetime, date
import threading
import queue
import numpy as np
import pandas as pd

import pytesseract
import os

TESSERACT_PATH = r"C:\Users\bbadner\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    print("[INFO] Using Tesseract at:", TESSERACT_PATH)
else:
    print("[ERROR] tesseract.exe not found at:", TESSERACT_PATH)

print("[INFO] Running motor_agent7 (full rewrite, Option A pending)")

# IMPORTANT: Use a non-GUI backend so Matplotlib works safely in worker threads
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fpdf import FPDF        # PDF report generation
import tkinter as tk         # GUI
from tkinter import ttk
import fitz                  # PyMuPDF for PDF parsing

# Optional OCR helpers (for scanned magnet PDFs)
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
REPORT_FOLDER_MOTOR  = "Motor Reports"
REPORT_FOLDER_MAGNET = "Magnet Reports"
LOGO_FILE = "rexair_logo.png"  # optional logo

# Magnet timing specification (for Cp/CpK)
MAGNET_TARGET = 3.7
MAGNET_TOL    = 0.5
LSL = MAGNET_TARGET - MAGNET_TOL
USL = MAGNET_TARGET + MAGNET_TOL

os.makedirs(REPORT_FOLDER_MOTOR,  exist_ok=True)
os.makedirs(REPORT_FOLDER_MAGNET, exist_ok=True)

# ============================================================
# Utility: Open File Cross-Platform
# ============================================================
def open_file(filepath):
    try:
        if platform.system() == "Windows":
            os.startfile(filepath)
        elif platform.system() == "Darwin":
            subprocess.run(["open", filepath])
        else:
            subprocess.run(["xdg-open", filepath])
    except Exception as e:
        print(f"[WARN] Could not open file: {e}")

# ============================================================
# OCR / PDF Text Extraction Helper
# ============================================================
def extract_pdf_text_with_ocr_fallback(full_path, ocr_language="eng"):
    """
    Extract text from a PDF.

    Pass 1: Native text extraction (fast, preferred)
    Pass 2: OCR via PyMuPDF (for scanned PDFs)

    Returns:
        text, mode
    mode ∈ {"text", "ocr", "failed_ocr:<err>", "read_error:<err>"}
    """
    # ---- Pass 1: native text
    try:
        doc = fitz.open(full_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        return "", f"read_error:{e}"

    if len(text.strip()) >= 50:
        return text, "text"

    # ---- Pass 2: OCR fallback
    try:
        doc = fitz.open(full_path)
        ocr_text = ""
        for page in doc:
            tp = page.get_textpage_ocr(language=ocr_language, dpi=300, full=True)
            ocr_text += page.get_text("text", textpage=tp)
        doc.close()
        return ocr_text, "ocr"
    except Exception as e:
        return text, f"failed_ocr:{e}"

# ============================================================
# Numeric Safety Helpers
# ============================================================
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
    if arr.size < 2:
        return 0.0
    return float(np.std(arr, ddof=1))


# ============================================================
# Part 2   Motor Filename Date Extraction
# ============================================================
def extract_date_from_filename(fname):
    """
    Extract date from filenames containing:
        C6521YYMMDD
    Example:
        C6521240517 -> 2024-05-17
    Returns:
        datetime or None
    """
    m = re.search(r"C6521(\d{6})", fname)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%y%m%d")
    except Exception:
        return None

# ============================================================
# Motor Excel Input Power Column Detection
# ============================================================
def find_input_power_column(df):
    """
    Locate the column that contains BOTH:
      - 'High Speed(Open)'
      - 'Input Power'
    within the same header cell or nearby.

    Fallback: any column containing 'Input Power'
    """
    # Pass 1: strict match
    for i in range(min(10, len(df))):
        row = df.iloc[i].astype(str).tolist()
        for j, cell in enumerate(row):
            cell_l = cell.lower()
            if "high speed(open)" in cell_l and "input power" in cell_l:
                return j

    # Pass 2: relaxed match
    for i in range(min(10, len(df))):
        row = df.iloc[i].astype(str).tolist()
        for j, cell in enumerate(row):
            if "input power" in cell.lower():
                return j

    return None

# ============================================================
# Motor Excel Reader
# ============================================================
def read_input_power(filepath):
    """
    Reads the motor test Excel file and extracts the
    Input Power values from the correct column.
    """
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
# Motor Diagnostics Overlay (text only)
# ============================================================
def add_motor_diagnostics(ax, labels, month_values):
    """
    Adds PASS/FAIL diagnostics text to a motor plot.
    Uses medians and standard deviations of per-file values.
    """
    medians = [safe_median(v) for v in month_values]
    stds    = [safe_stdev(v)  for v in month_values]

    def check(series_vals, ok_text, bad_text):
        if len(series_vals) < 2:
            return ok_text, "green"
        mu = float(np.mean(series_vals))
        sigma = float(np.std(series_vals, ddof=1))
        if sigma == 0:
            return ok_text, "green"
        current = float(series_vals[-1])
        return (ok_text, "green") if (mu - 3*sigma <= current <= mu + 3*sigma) else (bad_text, "red")

    last_12_medians = medians[-12:]
    last_3_medians  = medians[-3:]
    last_12_stds    = stds[-12:]
    last_3_stds     = stds[-3:]

    results = [
        check(last_12_medians, "12 Month Median is OK", "12 Month Median is Bad"),
        check(last_3_medians,  "3 Month Median is OK",  "3 Month Median is Bad"),
        check(last_12_stds,    "12 Month Std Dev is Good", "12 Month Std Dev is Bad"),
        check(last_3_stds,     "3 Month Std Dev is Good",  "3 Month Std Dev is Bad"),
    ]

    y_positions = [0.98, 0.92, 0.86, 0.80]
    for (text, color), y in zip(results, y_positions):
        ax.text(
            0.01, y, text,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            color=color,
            va="top"
        )

# ============================================================
#   Part 3
 #  Motor PDF Report Generation
#   OPTION A APPLIED HERE:
#   - Real dates used for plotting
#   - Labels used ONLY for tick display
# ============================================================
# ============================================================
# PART 3 — STATISTICS & METADATA UTILITIES
# ============================================================

def compute_cp_cpk(mean, std, lsl=-1.0, usl=1.0):
    """
    Compute Cp and Cpk given mean and standard deviation.
    Default spec limits are placeholders; adjust if needed.
    """
    if std is None or std <= 0:
        return None, None

    cp = (usl - lsl) / (6.0 * std)
    cpk = min(
        (usl - mean) / (3.0 * std),
        (mean - lsl) / (3.0 * std)
    )

    return cp, cpk


# ------------------------------------------------------------
# Extract mean/std from summary-style magnet PDFs (April/May)
# ------------------------------------------------------------
def extract_mean_std_cp_cpk(text):
    """
    Extract mean and standard deviation from magnet summary PDFs
    that already contain statistics (typically April / May).
    Returns dict with keys: mean, std
    """
    result = {
        "mean": None,
        "std": None
    }

    if not text:
        return result

    # Common summary patterns
    mean_match = re.search(
        r"Mean\s*[:=]\s*([-+]?\d*\.\d+|\d+)",
        text,
        re.IGNORECASE
    )

    std_match = re.search(
        r"(Std\s*Dev|Standard\s*Deviation)\s*[:=]\s*([-+]?\d*\.\d+|\d+)",
        text,
        re.IGNORECASE
    )

    try:
        if mean_match:
            result["mean"] = float(mean_match.group(1))

        if std_match:
            result["std"] = float(std_match.group(2))

    except Exception:
        pass

    return result


# ------------------------------------------------------------
# Extract month from PDF filename or metadata
# ------------------------------------------------------------
def extract_month_from_pdf_filename(filename, full_path=None):
    """
    Determine report month from filename or PDF metadata.
    Returns datetime object (first day of month).
    """

    # 1️⃣ Try YYYYMMDD in filename
    match = re.search(r"(20\d{2})(\d{2})\d{2}", filename)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        return datetime(year, month, 1)

    # 2️⃣ Try YYYY-MM in filename
    match = re.search(r"(20\d{2})[-_](\d{2})", filename)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        return datetime(year, month, 1)

    # 3️⃣ Try PDF metadata (creation date)
    try:
        if full_path:
            doc = fitz.open(full_path)
            meta = doc.metadata
            doc.close()

            if meta and "creationDate" in meta:
                m = re.search(
                    r"D:(\d{4})(\d{2})",
                    meta["creationDate"]
                )
                if m:
                    return datetime(
                        int(m.group(1)),
                        int(m.group(2)),
                        1
                    )
    except Exception:
        pass

    # 4️⃣ Fallback: current month
    return datetime.now().replace(day=1)


# ============================================================
# PART 4 — MAGNET HELPERS (OCR, PARSING, CAPABILITY)
# ============================================================

# ============================================================
# PART 4 — PDF TEXT & OCR EXTRACTION
# ============================================================

def extract_pdf_text_with_ocr_fallback(pdf_path):
    """
    Attempt to extract text from a PDF using PyMuPDF.
    Returns (text, mode) where mode is 'text' or 'ocr'.
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""

        for page in doc:
            text += page.get_text()

        doc.close()

        if text.strip():
            return text, "text"

    except Exception as e:
        print(f"[PDF] Text extraction failed: {e}")

    return "", "ocr"


# ------------------------------------------------------------
# OCR magnet timing values from scanned PDFs (AUTHORITATIVE)
# ------------------------------------------------------------
def extract_timing_values_from_pdf_images(pdf_path):
    """
    Extract numeric timing values from scanned magnet PDFs
    using aggressive OCR preprocessing.
    Returns a list of floats.
    """
    values = []

    try:
        doc = fitz.open(pdf_path)

        for page_index in range(len(doc)):
            page = doc.load_page(page_index)

            # ------------------------------------------------
            # Render page at HIGH DPI (critical for small text)
            # ------------------------------------------------
            pix = page.get_pixmap(dpi=400)
            img = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples
            )

            # ------------------------------------------------
            # Convert to grayscale
            # ------------------------------------------------
            img = img.convert("L")

            # ------------------------------------------------
            # Upscale image to thicken strokes
            # ------------------------------------------------
            scale = 2
            img = img.resize(
                (img.width * scale, img.height * scale),
                Image.BICUBIC
            )

            # ------------------------------------------------
            # Increase contrast
            # ------------------------------------------------
            img = ImageEnhance.Contrast(img).enhance(3.0)

            # ------------------------------------------------
            # Adaptive threshold (remove grid lines / noise)
            # ------------------------------------------------
            img_np = np.array(img)

            img_np = cv2.adaptiveThreshold(
                img_np,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                5
            )

            # ------------------------------------------------
            # OCR configuration (numbers only)
            # ------------------------------------------------
            config = (
                "--oem 3 "
                "--psm 6 "
                "-c tessedit_char_whitelist=0123456789.-"
            )

            text = pytesseract.image_to_string(
                img_np,
                config=config
            )

            # ------------------------------------------------
            # Extract floating-point numbers
            # ------------------------------------------------
            for token in re.findall(r"-?\d+\.\d+", text):
                try:
                    values.append(float(token))
                except ValueError:
                    pass

        doc.close()

    except Exception as e:
        print(f"[MAGNET][OCR] Exception: {e}")

    if values:
        print(
            f"[MAGNET][OCR] Extracted {len(values)} timing values from "
            f"{os.path.basename(pdf_path)}"
        )
    else:
        print(
            f"[MAGNET][OCR] No timing values found in "
            f"{os.path.basename(pdf_path)}"
        )

    return values



# ============================================================
# PART 5 — MAGNET ANALYSIS RUNNER
# Includes:
# - PDF processing
# - OCR + image OCR
# - Loose OCR reconstruction (January fix)
# - Excel fallback
# - Month labeling (January 1, January 2, etc.)
# ============================================================

# ============================================================
# PART 5 — MAGNET ANALYSIS RUNNER (FINAL FIXED VERSION)
# ============================================================

# ============================================================
# PART 5 — MAGNET ANALYSIS RUNNER (CLEAN & FINAL)
# ============================================================

def run_magnet_analysis(q=None, cancel_flag=None):
    base = os.path.dirname(os.path.abspath(
        sys.executable if getattr(sys, "frozen", False) else __file__
    ))

    report_dir = os.path.join(base, REPORT_FOLDER_MAGNET)
    os.makedirs(report_dir, exist_ok=True)

    rows = []
    MIN_VALUES = 10

    # ------------------------------------------------------------
    # Process magnet PDF files
    # ------------------------------------------------------------
    pdf_files = [
        f for f in os.listdir(base)
        if f.lower().endswith(".pdf")
    ]

    for f in pdf_files:
        if cancel_flag and cancel_flag.get("stop"):
            return

        full_path = os.path.join(base, f)

        # Best-effort text extraction (April / May PDFs)
        text, _ = extract_pdf_text_with_ocr_fallback(full_path)

        parsed = extract_mean_std_cp_cpk(text)
        mean = parsed["mean"]
        std  = parsed["std"]

        # --------------------------------------------------------
        # CASE 1 — Summary-stat PDFs (April / May)
        # --------------------------------------------------------
        if mean is not None and std is not None and std > 0:
            cp, cpk = compute_cp_cpk(mean, std)

        # --------------------------------------------------------
        # CASE 2 — Scanned PDFs (January) → OCR IS AUTHORITATIVE
        # --------------------------------------------------------
        else:
            ocr_vals = extract_timing_values_from_pdf_images(full_path)

            if len(ocr_vals) < MIN_VALUES:
                print(f"[MAGNET] SKIP (OCR failed): {f}")
                continue

            arr = np.array(ocr_vals, dtype=float)
            mean = float(arr.mean())
            std  = float(arr.std(ddof=1))
            cp, cpk = compute_cp_cpk(mean, std)

            print(f"[MAGNET] OCR USED AS SOURCE: {f}")

        # --------------------------------------------------------
        # Commit row (THIS WAS THE MISSING STEP BEFORE)
        # --------------------------------------------------------
        month_dt = extract_month_from_pdf_filename(f, full_path)

        rows.append({
            "month_dt": month_dt,
            "mean": mean,
            "std": std,
            "cp": cp,
            "cpk": cpk,
            "source_file": f
        })

        print(
            f"[MAGNET] OK: {f} "
            f"{month_dt.strftime('%Y-%m')} "
            f"mean={mean:.4f} std={std:.4f} cpk={cpk:.2f}"
        )

    if not rows:
        print("❌ No magnet data found.")
        return

    # ------------------------------------------------------------
    # Label files as "January 1", "January 2", etc.
    # ------------------------------------------------------------
    rows.sort(key=lambda r: (r["month_dt"], r["source_file"]))

    from collections import defaultdict
    month_counter = defaultdict(int)

    for r in rows:
        month_counter[r["month_dt"]] += 1
        r["label"] = (
            f"{r['month_dt'].strftime('%B')} "
            f"{month_counter[r['month_dt']]}"
        )

    filename = f"Rexair_Magnet_Timing_CpK_Report_{datetime.now():%Y-%m}.pdf"
    outpath = os.path.join(report_dir, filename)

    create_magnet_pdf(outpath, rows)
    print(f"[INFO] ✅ Magnet report created: {outpath}")
    open_file(outpath)


# ============================================================
# PART 6 — MAGNET PDF REPORT, GUI, THREADING, MAIN ENTRY
# ============================================================

# ------------------------------------------------------------
# Magnet PDF Report Generation
# ------------------------------------------------------------
def create_magnet_pdf(output_path, rows):
    """
    rows: list of dicts with keys:
      - month_dt
      - label
      - mean
      - std
      - cp
      - cpk
      - source_file
    """

    labels = [r["label"] for r in rows]
    cpks   = [r["cpk"] for r in rows]

    # ---- Create chart
    fig = plt.figure(figsize=(8.5, 4.0))
    plt.plot(labels, cpks, marker="o")
    plt.axhline(1.33, linestyle="--", linewidth=1.2)
    plt.title("Magnet Timing CpK Over Time")
    plt.xlabel("Inspection Batch")
    plt.ylabel("CpK")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    img_path = "magnet_cpk_plot_temp.png"
    plt.savefig(img_path, dpi=300)
    plt.close(fig)

    # ---- Build PDF
    pdf = FPDF()
    pdf.add_page()

    if os.path.exists(LOGO_FILE):
        pdf.image(LOGO_FILE, x=10, y=8, w=28)

    pdf.set_font("Arial", "B", 16)
    pdf.set_xy(45, 10)
    pdf.cell(0, 10, "Rexair Magnet Timing CpK Report", ln=True)

    pdf.image(img_path, x=10, y=30, w=190)

    pdf.set_xy(10, 125)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Summary Table", ln=True)

    pdf.set_font("Arial", "", 11)
    for r in rows:
        pdf.multi_cell(
            0,
            8,
            f"{r['label']} | "
            f"Mean: {r['mean']:.4f} | "
            f"Std: {r['std']:.4f} | "
            f"Cp: {r['cp']:.2f} | "
            f"CpK: {r['cpk']:.2f}"
        )

    pdf.output(output_path)

    try:
        if os.path.exists(img_path):
            os.remove(img_path)
    except Exception:
        pass


# ------------------------------------------------------------
# Threaded Runner Helper
# ------------------------------------------------------------
def start_threaded(funcs):
    root = tk.Tk()
    root.title("Processing...")
    root.geometry("420x120")

    tk.Label(
        root,
        text="Processing… please wait",
        font=("Segoe UI", 12)
    ).pack(pady=10)

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
                bar.stop()
                root.destroy()
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
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    root.after(200, check_done)
    root.mainloop()


# ------------------------------------------------------------
# GUI Launcher
# ------------------------------------------------------------
def launch_gui():
    root = tk.Tk()
    root.title("Motor & Magnet Analysis")
    root.geometry("520x300")

    tk.Label(
        root,
        text="Motor & Magnet Analysis",
        font=("Segoe UI", 14, "bold")
    ).pack(pady=12)

    tk.Button(
        root,
        text="Run Motor Analysis",
        width=45,
        command=lambda: start_threaded([run_motor_analysis])
    ).pack(pady=8)

    tk.Button(
        root,
        text="Run Magnet Analysis",
        width=45,
        command=lambda: start_threaded([run_magnet_analysis])
    ).pack(pady=8)

    tk.Button(
        root,
        text="Run Both",
        width=45,
        command=lambda: start_threaded(
            [run_motor_analysis, run_magnet_analysis]
        )
    ).pack(pady=8)

    root.mainloop()


# ------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    launch_gui()
