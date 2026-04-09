# ============================================================
# Motor & Magnet Analysis Agent (UPDATED - SAFE STDEV + CLEAN DIAGNOSTICS)
# ============================================================

import os
import re
import sys
import platform
import subprocess
from datetime import datetime
import threading
import queue
import calendar
import statistics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import tkinter as tk
from tkinter import ttk

import fitz  # PyMuPDF

# ============================================================
# Configuration
# ============================================================

REPORT_FOLDER_MOTOR = "Motor Reports"
REPORT_FOLDER_MAGNET = "Magnet Reports"
LOGO_FILE = "rexair_logo.png"  # must be in same directory, optional

# Spec limits for magnet timing
MAGNET_TARGET = 3.7
MAGNET_TOL = 0.5
LSL = MAGNET_TARGET - MAGNET_TOL
USL = MAGNET_TARGET + MAGNET_TOL

os.makedirs(REPORT_FOLDER_MOTOR, exist_ok=True)
os.makedirs(REPORT_FOLDER_MAGNET, exist_ok=True)

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
# Numeric Safety Helpers (FIXES YOUR ERROR)
# ============================================================

def safe_numeric_array(values):
    """
    Converts any iterable into a clean numpy float array.
    - coercing non-numeric to NaN then dropping
    - dropping inf
    """
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
    """
    Sample standard deviation (ddof=1).
    Returns 0.0 if <2 numeric samples.
    This avoids Python statistics.stdev() crash:
    AttributeError: 'float' object has no attribute 'numerator'
    """
    arr = safe_numeric_array(values)
    if arr.size < 2:
        return 0.0
    return float(np.std(arr, ddof=1))

# ============================================================
# Data Extract Helpers
# ============================================================

def extract_date_from_filename(fname):
    """
    Extract date from 'C6521YYMMDD' in filename.
    Example: C6521240517 -> 2024-05-17
    """
    m = re.search(r"C6521(\d{6})", fname)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%y%m%d")
    except Exception:
        return None

def find_input_power_column(df):
    """
    Finds a cell containing 'High Speed(Open)' within first 10 rows.
    Returns the column index that contains the marker.
    """
    target = "High Speed(Open)"
    for i in range(min(10, len(df))):
        row = df.iloc[i].astype(str).tolist()
        for j, cell in enumerate(row):
            if target in cell:
                return j
    return None

def read_input_power(filepath):
    """
    Reads motor excel and extracts numeric values from the column where
    'High Speed(Open)' marker is found (within first ~10 rows).
    """
    try:
        df = pd.read_excel(filepath, header=None)
    except Exception as e:
        print(f"[ERROR] Could not read excel: {filepath} -> {e}")
        return []

    col_idx = find_input_power_column(df)
    if col_idx is None:
        print(f"[WARN] Could not find 'High Speed(Open)' in {os.path.basename(filepath)}")
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
# Motor Diagnostics Overlay (PASS/FAIL TEXT ONLY)
# ============================================================

def add_motor_diagnostics(ax, months, month_values):
    """
    Your requested logic:
      - Remove all diagnostic lines (no clutter)
      - Determine +/- 3 sigma of monthly MEDIAN over past 12 months (including current),
        then check current month median: OK/BAD
      - Same for past 3 months median
      - Determine +/- 3 sigma of monthly STD DEV over past 12 months (including current),
        then check current month std dev: Good/Bad
      - Same for past 3 months std dev
      - Mean-of-medians and mean-of-std-devs
      - sigma==0 => OK/Good
      - Text top-left, bold, pass/fail only
    """

    # Build monthly stats aligned with plotted months
    medians = []
    stds = []
    for vals in month_values:
        medians.append(safe_median(vals))
        stds.append(safe_stdev(vals))

    dfm = pd.DataFrame({"Month": months, "Median": medians, "StdDev": stds}).dropna()
    dfm = dfm.sort_values("Month")

    def check(series, ok_text, bad_text):
        if len(series) < 2:
            return ok_text, "green"

        mu = float(series.mean())
        sigma = float(series.std(ddof=1))

        # Your rule: if sigma == 0, report OK/Good
        if sigma == 0:
            return ok_text, "green"

        lower = mu - 3.0 * sigma
        upper = mu + 3.0 * sigma
        current_val = float(series.iloc[-1])

        if lower <= current_val <= upper:
            return ok_text, "green"
        return bad_text, "red"

    last_12 = dfm.tail(12)
    last_3 = dfm.tail(3)

    results = [
        check(last_12["Median"], "12 Month Median is OK", "12 Month Median is Bad"),
        check(last_3["Median"],  "3 Month Median is OK",  "3 Month Median is Bad"),
        check(last_12["StdDev"], "12 Month Standard Deviation is Good", "12 Month Standard Deviation is Bad"),
        check(last_3["StdDev"],  "3 Month Standard Deviation is Good",  "3 Month Standard Deviation is Bad"),
    ]

    y_positions = [0.98, 0.92, 0.86, 0.80]
    for (txt, color), y in zip(results, y_positions):
        ax.text(
            0.01, y, txt,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            color=color,
            va="top"
        )

# ============================================================
# PDF: Motor Report
# ============================================================

def create_motor_pdf(output_path, months, month_values):
    pdf = FPDF()
    pdf.add_page()

    if os.path.exists(LOGO_FILE):
        pdf.image(LOGO_FILE, x=10, y=8, w=28)

    pdf.set_font("Arial", "B", 16)
    pdf.set_xy(45, 10)
    pdf.cell(0, 10, "Rexair Motor Test Trend Analysis", ln=True)

    # ------------- Boxplot -------------
    plt.figure(figsize=(8.5, 4.2))
    plt.boxplot(month_values, labels=[m.strftime("%b %Y") for m in months], showfliers=False)
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

    # ------------- Std Dev Trend -------------
    monthly_stds = [safe_stdev(v) for v in month_values]

    plt.figure(figsize=(8.5, 3.0))
    plt.plot([m.strftime("%b %Y") for m in months], monthly_stds, marker="o")
    plt.title("Standard Deviation Over Time")
    plt.ylabel("Std Dev")
    plt.grid(True)

    ax = plt.gca()
    add_motor_diagnostics(ax, months, month_values)

    plt.xticks(rotation=45)
    plt.tight_layout()
    img_std = "motor_std_temp.png"
    plt.savefig(img_std, dpi=300)
    plt.close()

    pdf.image(img_std, x=10, y=125, w=190)

    # ------------- Page 2 summary -------------
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

    # cleanup temp images
    for tmp in (img_box, img_std):
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

# ============================================================
# PDF: Magnet Report
# ============================================================

def create_magnet_pdf(output_path, rows):
    pdf = FPDF()
    pdf.add_page()

    if os.path.exists(LOGO_FILE):
        pdf.image(LOGO_FILE, x=10, y=8, w=28)

    pdf.set_font("Arial", "B", 16)
    pdf.set_xy(45, 10)
    pdf.cell(0, 10, "Rexair Magnet Timing CpK Report", ln=True)

    months = [r[0].strftime("%b %Y") for r in rows]
    cpks = [r[4] for r in rows]

    plt.figure(figsize=(8.5, 4.0))
    plt.plot(months, cpks, marker="o")
    plt.axhline(1.33, color="black", linestyle="--", linewidth=1.3)
    plt.title("Magnet Timing CpK Over Time")
    plt.xlabel("Month")
    plt.ylabel("CpK")
    plt.ylim(0, max(4, (max(cpks) if cpks else 4)))
    plt.grid(True)

    # Your rule: print Good/Bad across chart based on latest CpK
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

    # Summary table
    pdf.set_xy(10, 125)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Month Summary", ln=True)
    pdf.set_font("Arial", "", 11)

    for r in rows:
        d = r[0].strftime("%B %Y")
        pdf.cell(0, 8, f"{d} | Mean: {r[1]:.4f} | Std: {r[2]:.4f} | Cp: {r[3]:.2f} | CpK: {r[4]:.2f}", ln=True)

    pdf.output(output_path)

    try:
        if os.path.exists(img):
            os.remove(img)
    except Exception:
        pass

# ============================================================
# Motor Runner
# ============================================================

def run_motor_analysis(q=None, cancel_flag=None):
    base = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
    report_dir = os.path.join(base, REPORT_FOLDER_MOTOR)
    os.makedirs(report_dir, exist_ok=True)

    files = [f for f in os.listdir(base) if f.lower().endswith(".xlsx")]

    dated = []
    for f in files:
        d = extract_date_from_filename(f)
        if d:
            dated.append((d, f))

    dated.sort(key=lambda x: x[0])

    monthly = {}
    for d, f in dated:
        if cancel_flag and cancel_flag.get("stop"):
            return

        month_key = d.strftime("%Y-%m")
        vals = read_input_power(os.path.join(base, f))
        if vals:
            monthly.setdefault(month_key, []).extend(vals)

    if not monthly:
        print("❌ No motor Input Power data found.")
        return

    all_months = sorted([datetime.strptime(k, "%Y-%m") for k in monthly.keys()])
    recent_months = all_months[-12:]  # chart shows rolling 12 months
    month_values = [monthly[m.strftime("%Y-%m")] for m in recent_months]

    filename = f"Rexair_Motor_Test_Trend_Analysis_{datetime.now():%Y-%m}.pdf"
    outpath = os.path.join(report_dir, filename)

    create_motor_pdf(outpath, recent_months, month_values)
    print(f"[INFO] ✅ Motor report created: {outpath}")
    open_file(outpath)

# ============================================================
# Magnet Runner
# ============================================================
#
def extract_magnet_mean_std_from_text(text: str):
    """
    Tries multiple patterns to find Mean and Std Dev in the PDF text.
    Returns (mean, std) or (None, None).
    """
    # Normalize spacing
    t = re.sub(r"[ \t]+", " ", text)

    mean_patterns = [
        r"\bMean\s*[:=]\s*([-+]?\d*\.?\d+)",
        r"\bAverage\s*[:=]\s*([-+]?\d*\.?\d+)",
        r"\bX\s*bar\s*[:=]\s*([-+]?\d*\.?\d+)",
    ]

    std_patterns = [
        r"\bStd\s*Dev\s*[:=]\s*([-+]?\d*\.?\d+)",
        r"\bStdDev\s*[:=]\s*([-+]?\d*\.?\d+)",
        r"\bStd\s*[:=]\s*([-+]?\d*\.?\d+)",
        r"\bStdev\s*[:=]\s*([-+]?\d*\.?\d+)",
        r"\bStandard\s*Deviation\s*[:=]\s*([-+]?\d*\.?\d+)",
        r"\bSigma\s*[:=]\s*([-+]?\d*\.?\d+)",
        r"\bσ\s*[:=]\s*([-+]?\d*\.?\d+)",
    ]

    mean_val = None
    std_val = None

    for pat in mean_patterns:
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            mean_val = float(m.group(1))
            break

    for pat in std_patterns:
        s = re.search(pat, t, re.IGNORECASE)
        if s:
            std_val = float(s.group(1))
            break

    return mean_val, std_val
#

def run_magnet_analysis(q=None, cancel_flag=None):
    base = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
    report_dir = os.path.join(base, REPORT_FOLDER_MAGNET)
    os.makedirs(report_dir, exist_ok=True)

    files = [f for f in os.listdir(base) if f.lower().endswith(".pdf")]

    if not files:
        print("❌ No PDF files found in script folder for magnet analysis.")
        return

    results = []
    for f in files:
        if cancel_flag and cancel_flag.get("stop"):
            return

        full_path = os.path.join(base, f)

        # Date: prefer YYYYMMDD in filename; fallback to file modified time
        m = re.search(r"(\d{8})", f)
        if m:
            try:
                dt = datetime.strptime(m.group(1), "%Y%m%d").date()
            except Exception:
                dt = datetime.fromtimestamp(os.path.getmtime(full_path)).date()
        else:
            dt = datetime.fromtimestamp(os.path.getmtime(full_path)).date()

        # read text
        try:
            doc = fitz.open(full_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
        except Exception as e:
            print(f"[MAGNET] SKIP (can't read): {f} -> {e}")
            continue

        mean, std = extract_magnet_mean_std_from_text(text)

        if mean is None or std is None:
            # This is the #1 reason you saw "no output"
            print(f"[MAGNET] SKIP (mean/std not found): {f}")
            continue

        if std <= 0:
            print(f"[MAGNET] SKIP (std <= 0): {f} | mean={mean} std={std}")
            continue

        cp = (USL - LSL) / (6 * std)
        cpk = min((mean - LSL) / (3 * std), (USL - mean) / (3 * std))

        results.append([dt, mean, std, cp, cpk])
        print(f"[MAGNET] OK: {f} | date={dt} mean={mean:.4f} std={std:.4f} cpk={cpk:.2f}")

    if not results:
        print("❌ No magnet CpK data found. (All PDFs were skipped; see [MAGNET] SKIP reasons above.)")
        return

    results.sort(key=lambda x: x[0])

    filename = f"Rexair_Magnet_Timing_CpK_Report_{datetime.now():%Y-%m}.pdf"
    outpath = os.path.join(report_dir, filename)

    create_magnet_pdf(outpath, results)
    print(f"[INFO] ✅ Magnet report created: {outpath}")
    open_file(outpath)




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
            if q.get_nowait() == "done":
                root.destroy()
        except queue.Empty:
            root.after(200, check_done)

    def on_close():
        cancel_flag["stop"] = True
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    root.after(200, check_done)
    root.mainloop()

def launch_gui():
    root = tk.Tk()
    root.title("Motor & Magnet Analysis")
    root.geometry("520x280")

    tk.Label(root, text="Motor & Magnet Analysis", font=("Segoe UI", 14, "bold")).pack(pady=12)

    tk.Button(root, text="Run Motor Analysis", width=45,
              command=lambda: start_threaded([run_motor_analysis])).pack(pady=8)

    tk.Button(root, text="Run Magnet Analysis", width=45,
              command=lambda: start_threaded([run_magnet_analysis])).pack(pady=8)

    tk.Button(root, text="Run Both", width=45,
              command=lambda: start_threaded([run_motor_analysis, run_magnet_analysis])).pack(pady=8)

    root.mainloop()

if __name__ == "__main__":
    launch_gui()

