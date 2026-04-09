
# ============================================================
# Motor & Magnet Analysis Agent (UPDATED - PDF RAW & Excel Sheet 2)
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

from fpdf import FPDF        # PDF report
import tkinter as tk         # GUI
from tkinter import ttk
import fitz                  # PyMuPDF

# ============================================================
# Configuration
# ============================================================
REPORT_FOLDER_MOTOR = "Motor Reports"
REPORT_FOLDER_MAGNET = "Magnet Reports"
LOGO_FILE = "rexair_logo.png"  # optional

# Spec limits for magnet timing (used to compute Cp/CpK)
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
    target = "High Speed(Open)"
    for i in range(min(10, len(df))):
        row = df.iloc[i].astype(str).tolist()
        for j, cell in enumerate(row):
            if target in cell:
                return j
    return None

def read_input_power(filepath):
    try:
        # Use openpyxl engine explicitly
        df = pd.read_excel(filepath, header=None, engine="openpyxl")
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

    plt.figure(figsize=(8.5, 4.2))
    plt.boxplot(month_values,
                tick_labels=[m.strftime("%b %Y") for m in months],
                showfliers=False)
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

    pdf.set_xy(10, 125)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Month Summary Table", ln=True)
    pdf.set_font("Arial", "", 11)
    for r in rows:
        d = r["month_dt"].strftime("%B %Y")
        pdf.cell(0, 8, f"{d} "
                        f"Mean: {r['mean']:.4f} "
                        f"Std: {r['std']:.4f} "
                        f"Cp: {r['cp']:.2f} "
                        f"CpK: {r['cpk']:.2f}", ln=True)

    pdf.output(output_path)
    try:
        if os.path.exists(img):
            os.remove(img)
    except Exception:
        pass

# ============================================================
# Magnet Parsing (ONE LOT PDF -> use filename date month + parse Mean/Std)
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

def extract_month_from_pdf_filename(fname, full_path):
    """
    Prefer YYYYMMDD anywhere in filename. Fallback to modified date.
    Returns a date set to first day of that month.
    """
    m = re.search(r"(\d{8})", fname)
    if m:
        try:
            dt = datetime.strptime(m.group(1), "%Y%m%d").date()
            return dt.replace(day=1)
        except Exception:
            pass
    dt = datetime.fromtimestamp(os.path.getmtime(full_path)).date()
    return dt.replace(day=1)

def _to_float(s):
    # handles commas
    return float(str(s).replace(",", "").strip())

def extract_mean_std_cp_cpk(text: str):
    """
    Flexible parsing for many report formats:
      Mean 3.7280   / Mean: 3.7280 / Mean = 3.7280
      Std Dev 0.0708 / Standard Deviation: 0.0708
      Cp: 2.35
      Cpk: 2.22 (accept CpK/Cpk/CPK)
    Returns dict with any found values.
    """
    t = text.replace("\u00a0", " ")
    t = re.sub(r"[ \t]+", " ", t)

    # Allow colon, equals, or just whitespace between label and number
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
    cp_pats = [
        rf"\bCp\b\s*[:=]?\s*{number_pat}",
    ]
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
    cp = (USL - LSL) / (6.0 * std)
    cpk = min((mean - LSL) / (3.0 * std), (USL - mean) / (3.0 * std))
    return cp, cpk

# --- NEW: pull the 30 timing values under the "3.7 ± 0.5°" column from PDF text ---
def extract_timing_values_from_pdf(text: str):
    """
    Returns a list of floats, e.g., [3.80, 3.82, ...] for the '3.7 ± 0.5°' column.
    Strategy:
      * Find all angle tokens like '3.71°' in the document text.
      * Keep those in a plausible range (3.0–4.5) and take the first 30 that appear
        after the '3.7 ± 0.5°' header (if we can locate it), else take the first 30 overall.
    """
    t = text.replace("\u00a0", " ")

    # Locate the header if possible (accept +/- or ± and optional degree symbol)
    header = re.search(r"3\.7\s*(?:[±+\-\/]|(?:\+\/-))\s*0\.5\s*°?", t)
    start_idx = header.start() if header else 0

    # Collect values after header first
    vals = []
    for m in re.finditer(r"([3-4]\.\d{2})\s*°", t[start_idx:]):
        v = float(m.group(1))
        if 3.0 <= v <= 4.5:
            vals.append(v)

    # Fallback to whole doc if not enough
    if len(vals) < 30:
        vals = []
        for m in re.finditer(r"([3-4]\.\d{2})\s*°", t):
            v = float(m.group(1))
            if 3.0 <= v <= 4.5:
                vals.append(v)

    return vals[:30]

# --- NEW: read magnet timing from Excel Sheet 2 ---
def read_magnet_timing_from_excel(filepath):
    """
    Reads Sheet 2 and finds the column labeled with '3.7 +/- .5' (or '3.7 ± 0.5°'),
    then returns the first 30 numeric values under it (far-right column in your sheet).
    """
    try:
        df = pd.read_excel(filepath, sheet_name=1, header=None, engine="openpyxl")
    except Exception as e:
        print(f"[MAGNET] Could not read excel: {filepath} -> {e}")
        return []

    # Robust header match (accept +/- or ± and optional degree symbol)
    header_pat = re.compile(r"3\.7\s*(?:[±+\-\/]|(?:\+\/-))\s*0\.5\s*°?", re.IGNORECASE)

    # Find the column index of the header in the top rows
    col_idx = None
    for i in range(min(10, len(df))):
        row = df.iloc[i].astype(str).tolist()
        for j, cell in enumerate(row):
            if header_pat.search(cell):
                col_idx = j
                break
        if col_idx is not None:
            break

    # Fallback: use last non-empty column
    if col_idx is None:
        non_empty_cols = [c for c in range(df.shape[1]) if df.iloc[:, c].notna().sum() > 0]
        col_idx = non_empty_cols[-1] if non_empty_cols else None

    if col_idx is None:
        print(f"[MAGNET] Could not locate '3.7 +/- .5' column in {os.path.basename(filepath)}")
        return []

    # Collect numeric degree values (expect around 30)
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
# Magnet Runner (LOT PDFs + Excel Sheet 2)
# ============================================================
def run_magnet_analysis(q=None, cancel_flag=None):
    base = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
    report_dir = os.path.join(base, REPORT_FOLDER_MAGNET)
    os.makedirs(report_dir, exist_ok=True)

    rows = []

    # ---------- Process PDFs ----------
    pdf_files = [f for f in os.listdir(base) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("ℹ️ No PDF files found in script folder for magnet analysis.")

    for f in pdf_files:
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

        # Try label-based parsing first
        lbl = extract_mean_std_cp_cpk(text)
        mean = lbl["mean"]
        std = lbl["std"]
        cp = lbl["cp"]
        cpk = lbl["cpk"]

        # --- NEW fallback: compute from raw values in the "3.7 ± 0.5°" column ---
        if (mean is None or std is None) or (std is not None and std <= 0):
            raw_vals = extract_timing_values_from_pdf(text)
            if len(raw_vals) >= 20:
                arr = np.array(raw_vals, dtype=float)
                mean = float(arr.mean())
                std = float(arr.std(ddof=1))
                cp, cpk = compute_cp_cpk(mean, std)
            else:
                print(f"[MAGNET] SKIP (Mean/Std not found AND could not parse 30 values): {f}")
                dump_magnet_debug(base, f, text)
                continue

        month_dt = extract_month_from_pdf_filename(f, full_path)
        rows.append({
            "month_dt": month_dt,
            "mean": float(mean),
            "std": float(std),
            "cp": float(cp if cp is not None else compute_cp_cpk(mean, std)[0]),
            "cpk": float(cpk if cpk is not None else compute_cp_cpk(mean, std)[1]),
            "source_file": f
        })
        print(f"[MAGNET] OK(PDF): {f} {month_dt.strftime('%Y-%m')} "
              f"mean={mean:.4f} std={std:.4f} cp={rows[-1]['cp']:.2f} cpk={rows[-1]['cpk']:.2f}")

    # ---------- Process Excel (Sheet 2, far-right "3.7 +/- .5" column) ----------
    xlsx_files = [f for f in os.listdir(base) if f.lower().endswith(".xlsx")]
    for f in xlsx_files:
        if cancel_flag and cancel_flag.get("stop"):
            return
        full = os.path.join(base, f)
        vals30 = read_magnet_timing_from_excel(full)
        if len(vals30) >= 20:
            arr = np.array(vals30, dtype=float)
            mean = float(arr.mean())
            std = float(arr.std(ddof=1))
            cp, cpk = compute_cp_cpk(mean, std)
            month_dt = extract_month_from_pdf_filename(f, full)  # reuse month resolver; uses mtime as fallback
            rows.append({
                "month_dt": month_dt,
                "mean": mean,
                "std": std,
                "cp": cp,
                "cpk": cpk,
                "source_file": f
            })
            print(f"[MAGNET] OK(Excel): {f} {month_dt.strftime('%Y-%m')} "
                  f"mean={mean:.4f} std={std:.4f} cp={cp:.2f} cpk={cpk:.2f}")

    if not rows:
        print("❌ No magnet data found. Check: Magnet Debug text dumps or Excel Sheet 2 header.")
        return

    # Deduplicate by month: keep the latest file occurrence
    rows.sort(key=lambda r: (r["month_dt"], r["source_file"]))
    dedup = {}
    for r in rows:
        dedup[r["month_dt"]] = r
    final_rows = [dedup[k] for k in sorted(dedup.keys())]

    filename = f"Rexair_Magnet_Timing_CpK_Report_{datetime.now():%Y-%m}.pdf"
    outpath = os.path.join(report_dir, filename)
    create_magnet_pdf(outpath, final_rows)
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
