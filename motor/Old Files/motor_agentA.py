#=============================================================
#  Section 1
#=============================================================
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

# --- Matplotlib (thread-safe) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- PDF + GUI ---
from fpdf import FPDF
import tkinter as tk
from tkinter import ttk

# --- PDF parsing / OCR ---
import fitz
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
LOGO_FILE = "rexair_logo.png"

MAGNET_TARGET = 3.7
MAGNET_TOL = 0.5
LSL = MAGNET_TARGET - MAGNET_TOL
USL = MAGNET_TARGET + MAGNET_TOL

os.makedirs(REPORT_FOLDER_MOTOR, exist_ok=True)
os.makedirs(REPORT_FOLDER_MAGNET, exist_ok=True)

#=============================================================
#  Section 2
#=============================================================

def open_file(filepath):
    try:
        if platform.system() == "Windows":
            os.startfile(filepath)
        elif platform.system() == "Darwin":
            subprocess.run(["open", filepath])
        else:
            subprocess.run(["xdg-open", filepath])
    except Exception:
        pass


def safe_numeric_array(values):
    s = pd.to_numeric(pd.Series(values), errors="coerce")
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


def extract_motor_date(fname):
    m = re.search(r"C6521(\d{6})", fname)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%y%m%d")


def read_motor_input_power(filepath):
    try:
        df = pd.read_excel(filepath, header=None, engine="openpyxl")
    except Exception:
        return []

    col_idx = None
    for i in range(min(10, len(df))):
        for j, cell in enumerate(df.iloc[i].astype(str)):
            if "input power" in cell.lower():
                col_idx = j
                break
        if col_idx is not None:
            break

    if col_idx is None:
        return []

    values = []
    for v in df.iloc[:, col_idx]:
        try:
            fv = float(v)
            if np.isfinite(fv):
                values.append(fv)
        except Exception:
            pass
    return values


def create_motor_pdf(output_path, months, month_values):
    pdf = FPDF()
    pdf.add_page()

    if os.path.exists(LOGO_FILE):
        pdf.image(LOGO_FILE, x=10, y=8, w=28)

    pdf.set_font("Arial", "B", 16)
    pdf.set_xy(45, 10)
    pdf.cell(0, 10, "Rexair Motor Test Trend Analysis", ln=True)

    labels = [m.strftime("%b %Y") for m in months]

    plt.figure(figsize=(8.5, 4.2))
    plt.boxplot(month_values, tick_labels=labels, showfliers=False)
    plt.ylabel("Watts")
    plt.title("Input Power (W): Monthly Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    img1 = "motor_box.png"
    plt.savefig(img1, dpi=300)
    plt.close()
    pdf.image(img1, x=10, y=30, w=190)

    stds = [safe_stdev(v) for v in month_values]
    x = np.arange(len(labels))
    plt.figure(figsize=(8.5, 3.0))
    plt.plot(x, stds, marker="o")
    plt.xticks(x, labels, rotation=45)
    plt.ylabel("Std Dev")
    plt.title("Standard Deviation Over Time")
    plt.grid(True)
    plt.tight_layout()
    img2 = "motor_std.png"
    plt.savefig(img2, dpi=300)
    plt.close()
    pdf.image(img2, x=10, y=125, w=190)

    pdf.output(output_path)

    for f in (img1, img2):
        if os.path.exists(f):
            os.remove(f)


def run_motor_analysis():
    base = os.path.dirname(os.path.abspath(__file__))
    monthly = defaultdict(list)

    for f in os.listdir(base):
        if not f.lower().endswith(".xlsx"):
            continue
        d = extract_motor_date(f)
        if not d:
            continue
        vals = read_motor_input_power(os.path.join(base, f))
        if vals:
            monthly[d.strftime("%Y-%m")].extend(vals)

    if not monthly:
        print("[MOTOR] No motor data found")
        return

    months = sorted(datetime.strptime(k, "%Y-%m") for k in monthly)[-12:]
    values = [monthly[m.strftime("%Y-%m")] for m in months]

    outpath = os.path.join(
        REPORT_FOLDER_MOTOR,
        f"Rexair_Motor_Test_Trend_{datetime.now():%Y-%m}.pdf"
    )
    create_motor_pdf(outpath, months, values)
    open_file(outpath)

#=============================================================
#  Section 3
#=============================================================

# ================= MAGNET CODE (UNCHANGED) ==================

def extract_pdf_text_with_ocr_fallback(full_path, ocr_language="eng"):
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


def _to_float(s):
    return float(str(s).replace(",", "").strip())


def extract_mean_std_cp_cpk(text):
    t = text.replace("\u00a0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    number_pat = r"([-+]?\d[\d,]*\.?\d*)"

    mean_pats = [rf"\bMean\b\s*[:=]?\s*{number_pat}"]
    std_pats = [rf"\bStd\b.*?\s*{number_pat}"]
    cp_pats = [rf"\bCp\b\s*[:=]?\s*{number_pat}"]
    cpk_pats = [rf"\bCpK\b\s*[:=]?\s*{number_pat}"]

    out = {"mean": None, "std": None, "cp": None, "cpk": None}

    for pat in mean_pats:
        m = re.search(pat, t, re.I)
        if m:
            out["mean"] = _to_float(m.group(1))
            break

    for pat in std_pats:
        s = re.search(pat, t, re.I)
        if s:
            out["std"] = _to_float(s.group(1))
            break

    for pat in cp_pats:
        c = re.search(pat, t, re.I)
        if c:
            out["cp"] = _to_float(c.group(1))
            break

    for pat in cpk_pats:
        k = re.search(pat, t, re.I)
        if k:
            out["cpk"] = _to_float(k.group(1))
            break

    return out


def compute_cp_cpk(mean, std):
    if std is None or std <= 0:
        return 0.0, 0.0
    cp = (USL - LSL) / (6.0 * std)
    cpk = min((mean - LSL) / (3.0 * std),
              (USL - mean) / (3.0 * std))
    return cp, cpk


def extract_month_from_pdf_filename(fname, full_path):
    m = re.search(r"(\d{8})", fname)
    if m:
        try:
            dt = datetime.strptime(m.group(1), "%Y%m%d").date()
            return dt.replace(day=1)
        except Exception:
            pass
    dt = datetime.fromtimestamp(os.path.getmtime(full_path)).date()
    return dt.replace(day=1)


def create_magnet_pdf(output_path, rows):
    pdf = FPDF()
    pdf.add_page()

    if os.path.exists(LOGO_FILE):
        pdf.image(LOGO_FILE, x=10, y=8, w=28)

    pdf.set_font("Arial", "B", 16)
    pdf.set_xy(45, 10)
    pdf.cell(0, 10, "Rexair Magnet Timing CpK Report", ln=True)

    labels = [r["label"] for r in rows]
    cpks = [r["cpk"] for r in rows]

    plt.figure(figsize=(8.5, 4.0))
    plt.plot(labels, cpks, marker="o")
    plt.axhline(1.33, color="black", linestyle="--", linewidth=1.3)
    plt.title("Magnet Timing CpK (Per File)")
    plt.ylabel("CpK")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    img = "magnet_cpk_temp.png"
    plt.savefig(img, dpi=300)
    plt.close()
    pdf.image(img, x=10, y=30, w=190)

    pdf.add_page()
    pdf.set_font("Arial", "", 11)
    for r in rows:
        pdf.cell(
            0, 8,
            f"{r['label']}  Mean={r['mean']:.4f}  Std={r['std']:.4f}  "
            f"Cp={r['cp']:.2f}  CpK={r['cpk']:.2f}",
            ln=True
        )

    pdf.output(output_path)
    if os.path.exists(img):
        os.remove(img)


def run_magnet_analysis():
    base = os.path.dirname(os.path.abspath(__file__))
    rows = []

    pdf_files = [f for f in os.listdir(base) if f.lower().endswith(".pdf")]

    for f in pdf_files:
        full_path = os.path.join(base, f)
        text, mode = extract_pdf_text_with_ocr_fallback(full_path)

        if len(text.strip()) < 50:
            continue

        lbl = extract_mean_std_cp_cpk(text)
        mean = lbl["mean"]
        std = lbl["std"]
        cp = lbl["cp"]
        cpk = lbl["cpk"]

        if mean is None or std is None:
            continue

        if cp is None or cpk is None:
            cp, cpk = compute_cp_cpk(mean, std)

        month_dt = extract_month_from_pdf_filename(f, full_path)

        rows.append({
            "month_dt": month_dt,
            "mean": mean,
            "std": std,
            "cp": cp,
            "cpk": cpk,
            "source_file": f
        })

    if not rows:
        print("[MAGNET] No magnet data found")
        return

    rows.sort(key=lambda r: (r["month_dt"], r["source_file"]))
    counters = defaultdict(int)
    for r in rows:
        counters[r["month_dt"]] += 1
        r["label"] = f"{r['month_dt'].strftime('%B')} {counters[r['month_dt']]}"

    outpath = os.path.join(
        REPORT_FOLDER_MAGNET,
        f"Rexair_Magnet_Timing_CpK_Report_{datetime.now():%Y-%m}.pdf"
    )
    create_magnet_pdf(outpath, rows)
    open_file(outpath)

#=============================================================
#  Section 4
#=============================================================

def run_threaded(func):
    threading.Thread(target=func, daemon=True).start()


def run_both():
    run_motor_analysis()
    run_magnet_analysis()


def launch_gui():
    root = tk.Tk()
    root.title("Motor & Magnet Analysis")
    root.geometry("520x260")

    ttk.Label(
        root,
        text="Motor & Magnet Analysis",
        font=("Segoe UI", 14, "bold")
    ).pack(pady=12)

    ttk.Button(
        root,
        text="Run Motor Analysis",
        width=45,
        command=lambda: run_threaded(run_motor_analysis)
    ).pack(pady=6)

    ttk.Button(
        root,
        text="Run Magnet Analysis",
        width=45,
        command=lambda: run_threaded(run_magnet_analysis)
    ).pack(pady=6)

    ttk.Button(
        root,
        text="Run Both",
        width=45,
        command=lambda: run_threaded(run_both)
    ).pack(pady=6)

    root.mainloop()


if __name__ == "__main__":
    launch_gui()
