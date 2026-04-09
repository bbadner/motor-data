# ============================================================
# Motor & Magnet Analysis Agent
#  - Per-file labels: "January 1", "January 2", ...
#  - Magnet date parsing Option C (prefer YYYY-MM-DD, then YYYYMMDD, else mtime)
#  - Motor figure/labels alignment + scoped diagnostics overlay
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

print("[INFO] Running motor_agent8 (per-file labels + Option C date parsing)")

# Use non-GUI backend so Matplotlib works in worker threads
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fpdf import FPDF
import tkinter as tk
from tkinter import ttk
import fitz  # PyMuPDF

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
LOGO_FILE = "rexair_logo.png"  # optional

# Magnet timing spec for capability
MAGNET_TARGET = 3.7
MAGNET_TOL    = 0.5
LSL = MAGNET_TARGET - MAGNET_TOL
USL = MAGNET_TARGET + MAGNET_TOL

os.makedirs(REPORT_FOLDER_MOTOR,  exist_ok=True)
os.makedirs(REPORT_FOLDER_MAGNET, exist_ok=True)

# ============================================================
# PDF / OCR Helpers
# ============================================================
def extract_pdf_text_with_ocr_fallback(full_path, ocr_language="eng"):
    """
    Returns (text, mode) where mode ∈ {"text", "ocr", "failed_ocr:<err>", "read_error:<err>"}.
    """
    # Pass 1: native text
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

    # Pass 2: OCR via PyMuPDF (requires Tesseract installed)
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
# Utility: Open File
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
# Numeric Helpers
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
# Filename Date Extract (Motor Excel -> 'C6521YYMMDD')
# ============================================================
def extract_date_from_motor_filename(fname):
    """
    Extracts date from 'C6521YYMMDD' like 'C6521240517' -> 2024-05-17.
    Returns datetime or None.
    """
    m = re.search(r"C6521(\d{6})", fname)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%y%m%d")
    except Exception:
        return None

# ============================================================
# Motor Input Power column finder
# ============================================================
def find_input_power_column(df):
    # Must contain BOTH "High Speed(Open)" and "Input Power"
    for i in range(min(10, len(df))):
        row = df.iloc[i].astype(str).tolist()
        for j, cell in enumerate(row):
            c = cell.lower()
            if "high speed(open)" in c and "input power" in c:
                return j
    # Fallback: try just "Input Power"
    for i in range(min(10, len(df))):
        row = df.iloc[i].astype(str).tolist()
        for j, cell in enumerate(row):
            if "input power" in cell.lower():
                return j
    return None

def read_input_power(filepath):
    try:
        df = pd.read_excel(filepath, header=None, engine="openpyxl")
    except Exception as e:
        print(f"[ERROR] Could not read excel: {filepath} -> {e}")
        return []
    col_idx = find_input_power_column(df)
    if col_idx is None:
        print(f"[WARN] Could not find 'High Speed(Open)' and 'Input Power' in {os.path.basename(filepath)}")
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
# Motor Diagnostics Overlay (scoped to each figure)
# ============================================================
def add_motor_diagnostics(ax, series_values):
    """
    series_values: list of per-file medians or stds in chronological order.
    Draws overlay text on the given axes only.
    """
    def check(series_vals, ok_text, bad_text):
        if len(series_vals) < 2:
            return ok_text, "green"
        mu = float(np.mean(series_vals))
        sigma = float(np.std(series_vals, ddof=1))
        if sigma == 0:
            return ok_text, "green"
        lower = mu - 3.0 * sigma
        upper = mu + 3.0 * sigma
        current_val = float(series_vals[-1])
        return (ok_text, "green") if (lower <= current_val <= upper) else (bad_text, "red")

    last_12 = series_values[-12:]
    last_3  = series_values[-3:]

    msgs = [
        check(last_12, "12 Entry Trend is OK", "12 Entry Trend is Bad"),
        check(last_3,  "3 Entry Trend is OK",  "3 Entry Trend is Bad"),
    ]
    y_positions = [0.95, 0.88]
    for (txt, color), y in zip(msgs, y_positions):
        ax.text(0.01, y, txt, transform=ax.transAxes,
                fontsize=11, fontweight="bold", color=color, va="top")

# ============================================================
# PDF: Motor Report (per-file labels)
# ============================================================
def create_motor_pdf(output_path, labels, perfile_values, info_rows):
    """
    labels: list[str] e.g., "January 1", ...
    perfile_values: list[list[float]] input power values for each file
    info_rows: list[dict] with keys: {"label", "file", "file_date", "mean", "median", "std"}
    """
    # ---- Figure 1: boxplot
    fig1 = plt.figure(figsize=(8.5, 4.2))
    plt.boxplot(perfile_values, labels=labels, showfliers=False)
    plt.title("Input Power (W): Per-File Distribution")
    plt.ylabel("Watts")
    medians = [safe_median(v) for v in perfile_values]
    ax1 = plt.gca()
    add_motor_diagnostics(ax1, medians)
    plt.xticks(rotation=45)
    plt.tight_layout()
    img_box = "motor_boxplot_temp.png"
    plt.savefig(img_box, dpi=300)
    plt.close(fig1)

    # ---- Figure 2: std over time
    fig2 = plt.figure(figsize=(8.5, 3.0))
    stds = [safe_stdev(v) for v in perfile_values]
    plt.plot(labels, stds, marker="o")
    plt.title("Standard Deviation Over Time (Per File)")
    plt.ylabel("Std Dev")
    plt.grid(True)
    ax2 = plt.gca()
    add_motor_diagnostics(ax2, stds)
    plt.xticks(rotation=45)
    plt.tight_layout()
    img_std = "motor_std_temp.png"
    plt.savefig(img_std, dpi=300)
    plt.close(fig2)

    # ---- Build PDF
    pdf = FPDF()
    pdf.add_page()
    if os.path.exists(LOGO_FILE):
        pdf.image(LOGO_FILE, x=10, y=8, w=28)
    pdf.set_font("Arial", "B", 16)
    pdf.set_xy(45, 10)
    pdf.cell(0, 10, "Rexair Motor Test Trend Analysis", ln=True)
    pdf.image(img_box, x=10, y=30, w=190)

    # Page 2 (std chart)
    pdf.add_page()
    if os.path.exists(LOGO_FILE):
        pdf.image(LOGO_FILE, x=10, y=8, w=28)
    pdf.set_font("Arial", "B", 16)
    pdf.set_xy(45, 10)
    pdf.cell(0, 10, "Standard Deviation Trend (Per File)", ln=True)
    pdf.image(img_std, x=10, y=30, w=190)

    # Page 3 summary (with true dates)
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Per-File Summary (with file dates)", ln=True)
    pdf.set_font("Arial", "", 11)
    for row in info_rows:
        dt_txt = row["file_date"].strftime("%Y-%m-%d") if isinstance(row["file_date"], (datetime, date)) else str(row["file_date"])
        pdf.cell(0, 8,
                 f"{row['label']} | {dt_txt} | {os.path.basename(row['file'])}  "
                 f"Mean={row['mean']:.2f} W, Median={row['median']:.2f} W, StdDev={row['std']:.4f}",
                 ln=True)

    pdf.output(output_path)

    for tmp in (img_box, img_std):
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

# ============================================================
# PDF: Magnet Report (per-file labels)
# ============================================================
def create_magnet_pdf(output_path, rows):
    """
    rows: list[dict] with keys:
      month_dt: date (first of month)
      label:    "January 1", ...
      mean, std, cp, cpk
      file:     full path or name
      actual_dt: original file date (for summary line)
    """
    labels = [r["label"] for r in rows]
    cpks   = [r["cpk"]  for r in rows]

    # Chart
    fig = plt.figure(figsize=(8.5, 4.0))
    plt.plot(labels, cpks, marker="o")
    plt.axhline(1.33, color="black", linestyle="--", linewidth=1.3)
    plt.title("Magnet Timing CpK Over Time (Per File)")
    plt.xlabel("File within Month")
    plt.ylabel("CpK")
    plt.ylim(0, max(4, (max(cpks) if cpks else 4)))
    plt.grid(True)
    plt.xticks(rotation=60)
    plt.tight_layout()
    img = "magnet_cpk_temp.png"
    plt.savefig(img, dpi=300)
    plt.close(fig)

    # PDF
    pdf = FPDF()
    pdf.add_page()
    if os.path.exists(LOGO_FILE):
        pdf.image(LOGO_FILE, x=10, y=8, w=28)
    pdf.set_font("Arial", "B", 16)
    pdf.set_xy(45, 10)
    pdf.cell(0, 10, "Rexair Magnet Timing CpK Report", ln=True)
    pdf.image(img, x=10, y=30, w=190)

    pdf.set_xy(10, 125)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Per-File Summary Table", ln=True)
    pdf.set_font("Arial", "", 11)
    for r in rows:
        dt_txt = r["actual_dt"].strftime("%Y-%m-%d") if isinstance(r["actual_dt"], (datetime, date)) else str(r["actual_dt"])
        pdf.cell(0, 8,
                 f"{r['label']}  ({dt_txt})  Mean: {r['mean']:.4f}  Std: {r['std']:.4f}  "
                 f"Cp: {r['cp']:.2f}  CpK: {r['cpk']:.2f}",
                 ln=True)

    pdf.output(output_path)
    try:
        if os.path.exists(img):
            os.remove(img)
    except Exception:
        pass

# ============================================================
# Magnet Parsing Helpers (Option C date preference)
# ============================================================
def dump_magnet_debug(base_dir, pdf_filename, extracted_text):
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

def extract_month_from_any_filename(fname: str, full_path: str) -> date:
    """
    Option C (month value):
      1) Prefer YYYY-MM-DD or YYYY_MM_DD in filename.
      2) Else prefer YYYYMMDD.
      3) Else fallback to file mtime.
    Returns: date set to first of that month.
    """
    # 1) YYYY-MM-DD or YYYY_MM_DD
    m = re.search(r"(20\d{2})[-_](0[1-9]|1[0-2])[-_](0[1-9]|[12]\d|3[01])", fname)
    if m:
        y, mo, da = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(y, mo, 1)
        except Exception:
            pass

    # 2) YYYYMMDD
    m2 = re.search(r"(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])", fname)
    if m2:
        y, mo, da = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
        try:
            return date(y, mo, 1)
        except Exception:
            pass

    # 3) fallback to mtime
    dt = datetime.fromtimestamp(os.path.getmtime(full_path)).date()
    return date(dt.year, dt.month, 1)

def extract_actual_date_from_any_filename(fname: str, full_path: str) -> date:
    """
    Option C (actual specific date):
      1) Prefer YYYY-MM-DD / YYYY_MM_DD
      2) Else prefer YYYYMMDD
      3) Else fallback to mtime
    Returns: date(y, m, d)
    """
    m = re.search(r"(20\d{2})[-_](0[1-9]|1[0-2])[-_](0[1-9]|[12]\d|3[01])", fname)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m2 = re.search(r"(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])", fname)
    if m2:
        return date(int(m2.group(1)), int(m2.group(2)), int(m2.group(3)))
    return datetime.fromtimestamp(os.path.getmtime(full_path)).date()

def _to_float(s):
    return float(str(s).replace(",", "").strip())

def extract_mean_std_cp_cpk(text: str):
    t = text.replace("\u00a0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    number_pat = r"([-+]?\d[\d,]*\.?\d*)"
    mean_pats = [
        rf"\bMean\b\s*[:=]?\s*{number_pat}",
        rf"\bAverage\b\s*[:=]?\s*{number_pat}",
        rf"\bX\s*bar\b\s*[:=]?\s*{number_pat}",
        rf"\bXbar\b\s*[:=]?\s*{number_pat}",
    ]
    std_pats = [
        rf"\bStd\s*Dev\b\s*[:=]?\s*{number_pat}",
        rf"\bStdDev\b\s*[:=]?\s*{number_pat}",
        rf"\bStdev\b\s*[:=]?\s*{number_pat}",
        rf"\bStandard\s*Deviation\b\s*[:=]?\s*{number_pat}",
        rf"\bSigma\b\s*[:=]?\s*{number_pat}",
        rf"\bσ\b\s*[:=]?\s*{number_pat}",
    ]
    cp_pats  = [rf"\bCp\b\s*[:=]?\s*{number_pat}"]
    cpk_pats = [
        rf"\bCpK\b\s*[:=]?\s*{number_pat}",
        rf"\bCpk\b\s*[:=]?\s*{number_pat}",
        rf"\bCPK\b\s*[:=]?\s*{number_pat}",
    ]

    out = {"mean": None, "std": None, "cp": None, "cpk": None}
    for pat in mean_pats:
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            out["mean"] = _to_float(m.group(1)); break
    for pat in std_pats:
        s = re.search(pat, t, re.IGNORECASE)
        if s:
            out["std"] = _to_float(s.group(1)); break
    for pat in cp_pats:
        c = re.search(pat, t, re.IGNORECASE)
        if c:
            out["cp"] = _to_float(c.group(1)); break
    for pat in cpk_pats:
        k = re.search(pat, t, re.IGNORECASE)
        if k:
            out["cpk"] = _to_float(k.group(1)); break
    return out

def compute_cp_cpk(mean, std):
    if std is None or std <= 0:
        return 0.0, 0.0
    cp  = (USL - LSL) / (6.0 * std)
    cpk = min((mean - LSL) / (3.0 * std), (USL - mean) / (3.0 * std))
    return cp, cpk

# Fuzzy timing extractor for text/OCR
def _extract_timing_values_fuzzy(raw_text: str, target_count: int = 30):
    if not raw_text:
        return []
    t = raw_text
    # Normalize OCR quirks
    t = t.replace('O', '0').replace('o', '0')
    t = t.replace('S', '5')
    t = t.replace('I', '1').replace('l', '1')
    t = t.replace(',', '.').replace('·', '.').replace(':', '.')

    # Prefer after '3.7 +/- 0.5' header
    header = re.search(r"3\.7\s*(?:±|\+/-)\s*0\.5", t)
    scan_text = t[header.start():] if header else t

    vals = []
    for m in re.finditer(r"\b([34]\.[0-9]{1,2})\b", scan_text):
        try:
            v = float(m.group(1))
        except Exception:
            continue
        if 3.0 <= v <= 4.5:
            vals.append(v)
        if len(vals) >= target_count:
            return vals[:target_count]

    for m in re.finditer(r"\b([34][0-9]{2})\b", scan_text):
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

def extract_timing_values_from_pdf(text: str):
    return _extract_timing_values_fuzzy(text, target_count=30)

def extract_timing_values_from_pdf_images(pdf_path, dpi=450):
    """Image-based OCR fallback for scanned magnet PDFs.
    Uses multiple crops and OCR configs, then fuzzy-parses numeric values.
    Requires Tesseract + pytesseract + Pillow.
    """
    if not _HAS_TESSERACT:
        return []
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return []

    # OCR configs: whitelist digits/dot to reduce noise
    configs = [
        '--psm 6 -c tessedit_char_whitelist=0123456789.',
        '--psm 4 -c tessedit_char_whitelist=0123456789.',
        '--psm 11 -c tessedit_char_whitelist=0123456789.'
    ]
    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples).convert('L')
        w, h = img.size

        # Try several crop windows (layout varies)
        if w >= h:  # landscape-like
            boxes = [
                (0.55, 0.10, 0.95, 0.95),
                (0.60, 0.10, 0.98, 0.95),
                (0.50, 0.15, 0.95, 0.98),
            ]
        else:       # portrait-like
            boxes = [
                (0.55, 0.12, 0.92, 0.95),
                (0.60, 0.12, 0.98, 0.95),
                (0.50, 0.15, 0.95, 0.98),
            ]

        for (lx, ty, rx, by) in boxes:
            crop = img.crop((int(w*lx), int(h*ty), int(w*rx), int(h*by)))
            crop = ImageEnhance.Contrast(crop).enhance(3.2)
            crop = crop.point(lambda p: 255 if p > 185 else 0)

            for cfg in configs:
                try:
                    ocr_text = pytesseract.image_to_string(crop, config=cfg)
                except Exception:
                    continue
                vals = _extract_timing_values_fuzzy(ocr_text, target_count=30)
                if len(vals) >= 25:
                    doc.close()
                    return vals[:30]

    doc.close()
    return []

# ============================================================
# Motor Runner (FILES -> individual entries labeled "Month N")
# ============================================================
def run_motor_analysis(q=None, cancel_flag=None):
    base = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
    report_dir = os.path.join(base, REPORT_FOLDER_MOTOR)
    os.makedirs(report_dir, exist_ok=True)

    files = [f for f in os.listdir(base) if f.lower().endswith(".xlsx")]
    dated = []
    for f in files:
        d = extract_date_from_motor_filename(f)
        if d:
            dated.append((d, f))

    # chronological order
    dated.sort(key=lambda x: (x[0], x[1]))

    # Build per-file records
    records = []
    month_counts = {}   # "YYYY-MM" -> count
    for d, f in dated:
        if cancel_flag and cancel_flag.get("stop"):
            return
        vals = read_input_power(os.path.join(base, f))
        if not vals:
            continue
        month_key = d.strftime("%Y-%m")
        month_counts[month_key] = month_counts.get(month_key, 0) + 1
        label = f"{d.strftime('%B')} {month_counts[month_key]}"  # "January 1", "January 2", ...
        records.append((d.replace(day=1), label, vals, f, d))    # (month_dt, label, values, file, actual_dt)

    if not records:
        print("❌ No motor Input Power data found.")
        return

    # Keep last 12 entries overall
    records.sort(key=lambda r: (r[0], r[3]))
    recent = records[-12:]

    labels         = [r[1] for r in recent]
    perfile_values = [r[2] for r in recent]
    info_rows = []
    for (_, lbl, vals, f, actual_d) in recent:
        info_rows.append({
            "label": lbl,
            "file": os.path.join(base, f),
            "file_date": actual_d.date(),
            "mean": safe_mean(vals),
            "median": safe_median(vals),
            "std": safe_stdev(vals),
        })

    filename = f"Rexair_Motor_Test_Trend_Analysis_{datetime.now():%Y-%m}.pdf"
    outpath = os.path.join(report_dir, filename)
    create_motor_pdf(outpath, labels, perfile_values, info_rows)
    print(f"[INFO] ✅ Motor report created: {outpath}")
    open_file(outpath)

# ============================================================
# Magnet Runner (FILES -> individual entries labeled "Month N")
# ============================================================
def run_magnet_analysis(q=None, cancel_flag=None):
    base = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
    report_dir = os.path.join(base, REPORT_FOLDER_MAGNET)
    os.makedirs(report_dir, exist_ok=True)

    rows = []

    # ---------- PDFs ----------
    pdf_files = [f for f in os.listdir(base) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("ℹ️ No PDF files found in script folder for magnet analysis.")

    for f in pdf_files:
        if cancel_flag and cancel_flag.get("stop"):
            return
        full_path = os.path.join(base, f)

        text, mode = extract_pdf_text_with_ocr_fallback(full_path, ocr_language="eng")
        if mode.startswith("read_error"):
            print(f"[MAGNET] SKIP (can't read): {f} -> {mode}")
            continue
        if len(text.strip()) < 50:
            print(f"[MAGNET] SKIP (no extractable text even after OCR={mode}): {f}")
            dump_magnet_debug(base, f, text)
            continue
        if mode == "ocr":
            print(f"[MAGNET] OCR used for: {f}")

        lbls = extract_mean_std_cp_cpk(text)
        mean, std, cp, cpk = lbls["mean"], lbls["std"], lbls["cp"], lbls["cpk"]

        if (mean is None or std is None) or (std is not None and std <= 0):
            raw_vals = extract_timing_values_from_pdf(text)
            if len(raw_vals) < 20:
                img_vals = extract_timing_values_from_pdf_images(full_path)
                if len(img_vals) >= 20:
                    raw_vals = img_vals
                    print(f"[MAGNET] Image-OCR used for timing values: {f}")
            if len(raw_vals) >= 20:
                arr = np.array(raw_vals, dtype=float)
                mean = float(arr.mean())
                std  = float(arr.std(ddof=1))
                cp, cpk = compute_cp_cpk(mean, std)
            else:
                print(f"[MAGNET] SKIP (Mean/Std not found AND could not parse timing values): {f}")
                dump_magnet_debug(base, f, text)
                continue

        month_dt  = extract_month_from_any_filename(f, full_path)      # Option C month
        actual_dt = extract_actual_date_from_any_filename(f, full_path) # Option C actual date

        rows.append({
            "month_dt": month_dt,
            "actual_dt": actual_dt,
            "mean": float(mean),
            "std": float(std),
            "cp": float(cp if cp is not None else compute_cp_cpk(mean, std)[0]),
            "cpk": float(cpk if cpk is not None else compute_cp_cpk(mean, std)[1]),
            "file": full_path,
            "source_file": f,
        })
        print(f"[MAGNET] OK(PDF): {f} {month_dt.strftime('%Y-%m')} mean={mean:.4f} std={std:.4f} cp={rows[-1]['cp']:.2f} cpk={rows[-1]['cpk']:.2f}")

    # ---------- Excel (Sheet 2 timing) ----------
    xlsx_files = [f for f in os.listdir(base) if f.lower().endswith(".xlsx") and 'magnet' in f.lower()]
    for f in xlsx_files:
        if cancel_flag and cancel_flag.get("stop"):
            return
        full = os.path.join(base, f)
        vals30 = read_magnet_timing_from_excel(full)
        if len(vals30) >= 20:
            arr = np.array(vals30, dtype=float)
            mean = float(arr.mean())
            std  = float(arr.std(ddof=1))
            cp, cpk = compute_cp_cpk(mean, std)
            month_dt  = extract_month_from_any_filename(f, full)      # Option C month
            actual_dt = extract_actual_date_from_any_filename(f, full) # Option C actual date
            rows.append({
                "month_dt": month_dt,
                "actual_dt": actual_dt,
                "mean": mean,
                "std": std,
                "cp": cp,
                "cpk": cpk,
                "file": full,
                "source_file": f
            })
            print(f"[MAGNET] OK(Excel): {f} {month_dt.strftime('%Y-%m')} mean={mean:.4f} std={std:.4f} cp={cp:.2f} cpk={cpk:.2f}")

    if not rows:
        print("❌ No magnet data found. Check: Magnet Debug text dumps or Excel Sheet 2 header.")
        return

    # Keep ALL files; assign per-month counters for "January 1", "January 2", ...
    rows.sort(key=lambda r: (r["month_dt"], r["actual_dt"], r["source_file"]))
    from collections import defaultdict
    month_counters = defaultdict(int)
    for r in rows:
        month_counters[r["month_dt"]] += 1
        r["label"] = f"{r['month_dt'].strftime('%B')} {month_counters[r['month_dt']]}"

    filename = f"Rexair_Magnet_Timing_CpK_Report_{datetime.now():%Y-%m}.pdf"
    outpath = os.path.join(report_dir, filename)
    create_magnet_pdf(outpath, rows)
    print(f"[INFO] Magnet report created: {outpath}")
    open_file(outpath)

# ============================================================
# Magnet Excel timing reader (Sheet 2)
# ============================================================
def read_magnet_timing_from_excel(filepath):
    try:
        df = pd.read_excel(filepath, sheet_name=1, header=None, engine="openpyxl")
    except Exception as e:
        print(f"[MAGNET] Could not read excel: {filepath} -> {e}")
        return []

    header_pat = re.compile(r"3\.7\s*(?:±|\+/-)\s*0\.5\s*°?", re.IGNORECASE)

    col_idx = None
    for i in range(min(10, len(df))):
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
        print(f"[MAGNET] Could not locate '3.7 +/- .5' column in {os.path.basename(filepath)}")
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

# ============================================================
# GUI / Threading
# ============================================================
def start_threaded(funcs):
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
                try: bar.stop()
                except Exception: pass
                try: root.destroy()
                except Exception: pass
                return
        except queue.Empty:
            pass
        root.after(200, check_done)

    def on_close():
        cancel_flag["stop"] = True
        try: bar.stop()
        except Exception: pass
        try: root.destroy()
        except Exception: pass

    root.protocol("WM_DELETE_WINDOW", on_close)
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    root.after(200, check_done)
    root.mainloop()

def launch_gui():
    root = tk.Tk()
    root.title("Motor & Magnet Analysis")
    root.geometry("520x300")
    tk.Label(root, text="Motor & Magnet Analysis", font=("Segoe UI", 14, "bold")).pack(pady=12)

    tk.Button(root, text="Run Motor Analysis", width=45,
              command=lambda: start_threaded([run_motor_analysis])).pack(pady=8)

    tk.Button(root, text="Run Magnet Analysis (PDF + Excel)", width=45,
              command=lambda: start_threaded([run_magnet_analysis])).pack(pady=8)

    tk.Button(root, text="Run Both", width=45,
              command=lambda: start_threaded([run_motor_analysis, run_magnet_analysis])).pack(pady=8)
    root.mainloop()

if __name__ == "__main__":
    launch_gui()