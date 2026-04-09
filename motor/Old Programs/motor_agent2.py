# ============================================================
# Motor & Magnet Analysis Agent (UPDATED - CORRECT MAGNET PARSING)
# ============================================================

import os
import re
import sys
import platform
import subprocess
from datetime import datetime
import threading
import queue

import numpy as np
import pandas as pd

# IMPORTANT: Use a non-GUI backend so Matplotlib works safely in worker threads
import matplotlib
matplotlib.use("Agg")
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
LOGO_FILE = "rexair_logo.png"  # optional

# Magnet timing spec limits (used if Cp/CpK not provided in source)
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
# Data Extract Helpers (Motor)
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
    - 12mo median ±3σ: OK/Bad (mean of medians)
    - 3mo  median ±3σ: OK/Bad
    - 12mo stddev ±3σ: Good/Bad (mean of stddevs)
    - 3mo  stddev ±3σ: Good/Bad
    - σ==0 => OK/Good
    - Text only, top-left, bold
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

    # Boxplot
    plt.figure(figsize=(8.5, 4.2))
    plt.boxplot(
        month_values,
        tick_labels=[m.strftime("%b %Y") for m in months],
        showfliers=False
    )
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

    # Std dev trend
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

    # Page 2 summary
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

    months = [r["month_dt"].strftime("%b %Y") for r in rows]
    cpks = [r["cpk"] for r in rows]

    plt.figure(figsize=(8.5, 4.0))
    plt.plot(months, cpks, marker="o")
    plt.axhline(1.33, color="black", linestyle="--", linewidth=1.3)
    plt.title("Magnet Timing CpK Over Time")
    plt.xlabel("Month")
    plt.ylabel("CpK")
    plt.ylim(0, max(4, (max(cpks) if cpks else 4)))
    plt.grid(True)

    # CpK Good/Bad overlay based on latest CpK
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
        d = r["month_dt"].strftime("%B %Y")
        pdf.cell(
            0, 8,
            f"{d} | Mean: {r['mean']:.4f} | Std: {r['std']:.4f} | Cp: {r['cp']:.2f} | CpK: {r['cpk']:.2f}",
            ln=True
        )

    pdf.output(output_path)

    try:
        if os.path.exists(img):
            os.remove(img)
    except Exception:
        pass

# ============================================================
# Magnet Extraction (CORRECT: parse month summary lines)
# ============================================================

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
}

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

def parse_month_summary_rows(text: str):
    """
    Extract lines like:
      April 2025 | Mean: 3.7280 | Std: 0.0708 | Cp: 2.35 | CpK: 2.22
    Returns list of dict rows with month_dt, mean, std, cp, cpk.
    """
    t = re.sub(r"[ \t]+", " ", text)

    # Accept CpK / Cpk / CPK and separators | or spaces
    pattern = re.compile(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})"
        r"\s*\|\s*Mean:\s*([-+]?\d*\.?\d+)"
        r"\s*\|\s*Std:\s*([-+]?\d*\.?\d+)"
        r"(?:\s*\|\s*Cp:\s*([-+]?\d*\.?\d+))?"
        r"(?:\s*\|\s*CpK:\s*([-+]?\d*\.?\d+)|\s*\|\s*Cpk:\s*([-+]?\d*\.?\d+)|\s*\|\s*CPK:\s*([-+]?\d*\.?\d+))?",
        re.IGNORECASE
    )

    rows = []
    for m in pattern.finditer(t):
        mon_name = m.group(1)
        year = int(m.group(2))
        mean = float(m.group(3))
        std = float(m.group(4))

        cp = m.group(5)
        cp = float(cp) if cp is not None else None

        # CpK might be in group 6,7,8 depending on capitalization match
        cpk = m.group(6) or m.group(7) or m.group(8)
        cpk = float(cpk) if cpk is not None else None

        month_num = _MONTHS[mon_name.lower()]
        month_dt = datetime(year, month_num, 1).date()

        rows.append({"month_dt": month_dt, "mean": mean, "std": std, "cp": cp, "cpk": cpk})

    return rows

def compute_cp_cpk(mean, std):
    if std <= 0:
        return 0.0, 0.0
    cp = (USL - LSL) / (6.0 * std)
    cpk = min((mean - LSL) / (3.0 * std), (USL - mean) / (3.0 * std))
    return cp, cpk

# ============================================================
# Magnet Runner
# ============================================================

def run_magnet_analysis(q=None, cancel_flag=None):
    base = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
    report_dir = os.path.join(base, REPORT_FOLDER_MAGNET)
    os.makedirs(report_dir, exist_ok=True)

    files = [f for f in os.listdir(base) if f.lower().endswith(".pdf")]

    if not files:
        print("❌ No PDF files found in script folder for magnet analysis.")
        return

    all_rows = []

    for f in files:
        if cancel_flag and cancel_flag.get("stop"):
            return

        full_path = os.path.join(base, f)

        # Read text
        try:
            doc = fitz.open(full_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
        except Exception as e:
            print(f"[MAGNET] SKIP (can't read): {f} -> {e}")
            continue

        if len(text.strip()) < 50:
            print(f"[MAGNET] SKIP (likely scanned/image PDF: no extractable text): {f}")
            dump_magnet_debug(base, f, text)
            continue

        rows = parse_month_summary_rows(text)

        if not rows:
            print(f"[MAGNET] SKIP (no Month Summary rows found): {f}")
            dump_magnet_debug(base, f, text)
            continue

        # Fill Cp/CpK if missing
        for r in rows:
            if r["cp"] is None or r["cpk"] is None:
                cp, cpk = compute_cp_cpk(r["mean"], r["std"])
                r["cp"] = cp if r["cp"] is None else r["cp"]
                r["cpk"] = cpk if r["cpk"] is None else r["cpk"]

        all_rows.extend(rows)
        print(f"[MAGNET] OK: {f} -> extracted {len(rows)} month row(s)")

    if not all_rows:
        print("❌ No magnet data found. Check: Magnet Debug text dumps.")
        return

    # Combine duplicates by month (if multiple PDFs contain the same month)
    # Keep the last occurrence (often latest report)
    all_rows.sort(key=lambda r: r["month_dt"])
    dedup = {}
    for r in all_rows:
        dedup[r["month_dt"]] = r
    final_rows = [dedup[k] for k in sorted(dedup.keys())]

    filename = f"Rexair_Magnet_Timing_CpK_Report_{datetime.now():%Y-%m}.pdf"
    outpath = os.path.join(report_dir, filename)

    create_magnet_pdf(outpath, final_rows)
    print(f"[INFO] ✅ Magnet report created: {outpath}")
    open_file(outpath)

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
    recent_months = all_months[-12:]
    month_values = [monthly[m.strftime("%Y-%m")] for m in recent_months]

    filename = f"Rexair_Motor_Test_Trend_Analysis_{datetime.now():%Y-%m}.pdf"
    outpath = os.path.join(report_dir, filename)

    create_motor_pdf(outpath, recent_months, month_values)
    print(f"[INFO] ✅ Motor report created: {outpath}")
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

