# ============================================================
# Rexair Motor & Magnet Analysis – CLEAN SCRIPT v13 (UPDATED)
# ============================================================

import os
import re
import platform
import subprocess
from datetime import datetime
import threading
import queue
import unicodedata

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fpdf import FPDF
import tkinter as tk
from tkinter import ttk, messagebox

import fitz  # PyMuPDF


REPORT_FOLDER_MOTOR = "Motor Reports"
REPORT_FOLDER_MAGNET = "Magnet Reports"

MAGNET_TARGET = 3.7
MAGNET_TOL = 0.5
LSL = MAGNET_TARGET - MAGNET_TOL
USL = MAGNET_TARGET + MAGNET_TOL

os.makedirs(REPORT_FOLDER_MOTOR, exist_ok=True)
os.makedirs(REPORT_FOLDER_MAGNET, exist_ok=True)

print("RUNNING CLEAN SCRIPT v13")


# ============================================================
# UTILITIES
# ============================================================

def open_file(filepath: str) -> None:
    """Open a file in the OS default viewer."""
    try:
        if platform.system() == "Windows":
            os.startfile(filepath)  # type: ignore[attr-defined]
        elif platform.system() == "Darwin":
            subprocess.run(["open", filepath], check=False)
        else:
            subprocess.run(["xdg-open", filepath], check=False)
    except Exception:
        pass


def versioned_filename(folder: str, name: str) -> str:
    """Create a versioned filename if one already exists."""
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        return path

    base, ext = os.path.splitext(name)
    i = 1
    while True:
        candidate = os.path.join(folder, f"{base}_v{i}{ext}")
        if not os.path.exists(candidate):
            return candidate
        i += 1


def compute_cp_cpk(mean: float, std: float) -> tuple[float, float]:
    """Compute Cp and Cpk using global LSL/USL."""
    if std <= 0:
        return 0.0, 0.0
    cp = (USL - LSL) / (6.0 * std)
    cpk = min((mean - LSL) / (3.0 * std), (USL - mean) / (3.0 * std))
    return float(cp), float(cpk)


def sanitize_latin1(text: str) -> str:
    """
    FPDF core fonts are latin-1 only. This removes/normalizes characters that
    would crash pdf.output(), such as FULLWIDTH parentheses.
    """
    if text is None:
        return ""
    # Normalize (turns many fullwidth / accented variations into simpler forms)
    norm = unicodedata.normalize("NFKD", str(text))
    # Replace common fullwidth punctuation explicitly
    norm = norm.replace("（", "(").replace("）", ")").replace("，", ",")
    # Strip anything not representable in latin-1
    return norm.encode("latin-1", "ignore").decode("latin-1")


def extract_date_anywhere(name: str) -> datetime | None:
    """
    Try to extract a date from filenames like:
      ...20250414...
      ...250414...
    Returns datetime or None.
    """
    s = sanitize_latin1(name)

    m = re.search(r"(20\d{6})", s)  # YYYYMMDD
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d")
        except Exception:
            pass

    m = re.search(r"(?<!\d)(\d{6})(?!\d)", s)  # YYMMDD (standalone)
    if m:
        try:
            return datetime.strptime(m.group(1), "%y%m%d")
        except Exception:
            pass

    return None


def pretty_magnet_label(filename: str, month_index: dict[str, int]) -> str:
    """
    Create labels like:
      Apr 1, May 1, Jan 1, Jan 2
    based on the date in filename (if present).
    Falls back to sanitized basename.
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    d = extract_date_anywhere(base)
    if not d:
        return sanitize_latin1(base)

    key = d.strftime("%Y-%m")
    month_index.setdefault(key, 0)
    month_index[key] += 1

    # Example: "Apr 1"
    return sanitize_latin1(f"{d.strftime('%b')} {month_index[key]}")


# ============================================================
# MOTOR SECTION
# ============================================================

def extract_motor_date_from_filename(fname: str) -> datetime | None:
    """
    Original logic expected "C6521YYMMDD" somewhere in the filename.
    """
    m = re.search(r"C6521(\d{6})", fname)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%y%m%d")
    except Exception:
        return None


def read_input_power(path: str) -> list[float]:
    """
    Read all numeric values between 500 and 900 from the excel file.
    (Keeps your original behavior.)
    """
    try:
        df = pd.read_excel(path, header=None)
    except Exception:
        return []

    vals: list[float] = []
    for v in df.values.flatten():
        try:
            x = float(v)
            if 500 < x < 900:
                vals.append(x)
        except Exception:
            pass

    return vals


def create_motor_pdf(out: str, months_dt: list[datetime], month_values: list[list[float]]) -> None:
    labels = [m.strftime("%b %Y") for m in months_dt]

    # BOX PLOT
    plt.figure(figsize=(11, 5))
    plt.boxplot(month_values, tick_labels=labels, showfliers=False)
    plt.xticks(rotation=45)
    plt.title("Motor Input Power – Monthly Distribution (Last 12 Months)")
    plt.tight_layout()
    box = "motor_box.png"
    plt.savefig(box, dpi=300)
    plt.close()

    # STD TREND
    stds = [np.std(v, ddof=1) if len(v) >= 2 else 0 for v in month_values]
    plt.figure(figsize=(11, 4))
    plt.plot(range(1, len(stds) + 1), stds, marker="o")
    plt.xticks(range(1, len(labels) + 1), labels, rotation=45)
    plt.title("Motor Input Power – Std Deviation Trend")
    plt.tight_layout()
    stdimg = "motor_std.png"
    plt.savefig(stdimg, dpi=300)
    plt.close()

    # PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Rexair Motor Test Trend Analysis", ln=True)
    pdf.image(box, x=10, y=30, w=190)

    pdf.add_page()
    pdf.image(stdimg, x=10, y=30, w=190)

    pdf.output(out)

    for p in (box, stdimg):
        try:
            os.remove(p)
        except Exception:
            pass


def run_motor_analysis(q=None, cancel_flag=None) -> None:
    base = os.getcwd()

    files = [
        f for f in os.listdir(base)
        if f.lower().endswith((".xlsx", ".xls")) and "motor test data" in f.lower()
    ]

    dated: list[tuple[datetime, str]] = []
    for f in files:
        d = extract_motor_date_from_filename(f)
        if d:
            dated.append((d, f))

    dated.sort(key=lambda x: x[0])

    monthly: dict[str, list[float]] = {}
    for d, f in dated:
        vals = read_input_power(os.path.join(base, f))
        if vals:
            monthly.setdefault(d.strftime("%Y-%m"), []).extend(vals)

    if not monthly:
        print("No motor data found")
        return

    months = sorted(monthly.keys())
    months_dt = [datetime.strptime(m, "%Y-%m") for m in months]
    if len(months_dt) > 12:
        months_dt = months_dt[-12:]

    month_values = [monthly[m.strftime("%Y-%m")] for m in months_dt]

    print("[MOTOR] Months used:", [m.strftime("%Y-%m") for m in months_dt])

    name = f"Rexair_Motor_Test_Trend_Analysis_{datetime.now():%Y-%m}.pdf"
    out = versioned_filename(REPORT_FOLDER_MOTOR, name)

    create_motor_pdf(out, months_dt, month_values)

    print("Motor report created:", out)
    open_file(out)


# ============================================================
# MAGNET SECTION
# ============================================================

def extract_magnet_values(pdf_path: str) -> list[float]:
    """
    Extract numeric values in expected range (3.4–4.0) from PDF text.
    Keeps your approach, but slightly safer parsing.
    """
    values: list[float] = []
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            text = page.get_text() or ""
            # Grab floats like 3.65 etc.
            for m in re.findall(r"\b\d+\.\d+\b", text):
                try:
                    v = float(m)
                    if 3.4 <= v <= 4.0:
                        values.append(v)
                except Exception:
                    pass
    finally:
        doc.close()

    if len(values) >= 10:
        print(f"[MAGNET] {os.path.basename(pdf_path)} -> {len(values)} values")
        return values

    return []


def create_magnet_pdf(output_path: str, results: list[dict]) -> None:
    """
    Generates:
      Page 1: CpK trend (PER FILE)
      Page 2: Histogram (ALL values combined)
      Page 3: Boxplot (ALL values combined)
      Page 4: Per-File Summary Table
    """
    # Labels like "Apr 1", "May 1", etc.
    month_index: dict[str, int] = {}

    labels: list[str] = []
    cpks: list[float] = []

    for r in results:
        label = pretty_magnet_label(r["file"], month_index)
        labels.append(label)
        cpks.append(float(r["cpk"]))

    # ---------------------------
    # CpK TREND (PER FILE)
    # ---------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(labels, cpks, marker="o", linewidth=2)
    plt.axhline(1.33, linestyle="--")
    plt.ylim(0, 4)
    plt.ylabel("Cpk", fontsize=12)
    plt.xlabel("File within Month", fontsize=12)
    plt.title("Magnet Timing CpK Over Time (Per File)", fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True)

    status = "CpK is Good" if min(cpks) >= 1.33 else "CpK is BAD"
    color = "green" if min(cpks) >= 1.33 else "red"
    plt.text(len(labels) / 2, 2, status, fontsize=26, color=color, ha="center")

    trend_img = "magnet_cpk_trend.png"
    plt.tight_layout()
    plt.savefig(trend_img, dpi=300)
    plt.close()

    # ---------------------------
    # HISTOGRAM (ALL VALUES)
    # ---------------------------
    all_values: list[float] = []
    for r in results:
        all_values.extend(r["values"])
    arr = np.array(all_values, dtype=float)

    plt.figure(figsize=(10, 5))
    plt.hist(arr, bins=12)
    plt.axvline(MAGNET_TARGET, linestyle="--")
    plt.title("Magnet Timing Histogram")
    hist_img = "magnet_hist.png"
    plt.tight_layout()
    plt.savefig(hist_img, dpi=300)
    plt.close()

    # ---------------------------
    # BOX PLOT (ALL VALUES)
    # ---------------------------
    plt.figure(figsize=(10, 4))
    plt.boxplot(arr, vert=False)
    plt.axvline(MAGNET_TARGET)
    plt.title("Magnet Timing Distribution")
    box_img = "magnet_box.png"
    plt.tight_layout()
    plt.savefig(box_img, dpi=300)
    plt.close()

    # ---------------------------
    # PDF BUILD
    # ---------------------------
    pdf = FPDF()

    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Rexair Magnet Timing CpK Report", ln=True)
    pdf.image(trend_img, x=10, y=40, w=180)

    pdf.add_page()
    pdf.image(hist_img, x=10, y=30, w=180)

    pdf.add_page()
    pdf.image(box_img, x=10, y=30, w=180)

    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Per-File Summary Table", ln=True)

    pdf.set_font("Arial", "", 11)
    for i, r in enumerate(results):
        # sanitize EVERYTHING that might include unicode
        label = sanitize_latin1(labels[i])
        filename_safe = sanitize_latin1(os.path.basename(r["file"]))

        mean = float(r["mean"])
        std = float(r["std"])
        cp, cpk = compute_cp_cpk(mean, std)

        line = f"{label}  ({filename_safe})  Mean: {mean:.4f}  Std: {std:.4f}  Cp: {cp:.2f}  CpK: {cpk:.2f}"
        pdf.cell(0, 8, sanitize_latin1(line), ln=True)

    pdf.output(output_path)

    for p in (trend_img, hist_img, box_img):
        try:
            os.remove(p)
        except Exception:
            pass


def run_magnet_analysis(q=None, cancel_flag=None) -> None:
    base = os.getcwd()

    # Heuristic: only scan PDFs that look like magnet PDFs
    pdfs = [
        f for f in os.listdir(base)
        if f.lower().endswith(".pdf")
        and ("magnet" in f.lower() or "checking" in f.lower())
    ]

    # If none match, fallback to all PDFs (keeps original behavior)
    if not pdfs:
        pdfs = [f for f in os.listdir(base) if f.lower().endswith(".pdf")]

    results: list[dict] = []

    for f in pdfs:
        full = os.path.join(base, f)
        vals = extract_magnet_values(full)
        if len(vals) < 10:
            continue

        arr = np.array(vals, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if len(arr) >= 2 else 0.0
        cp, cpk = compute_cp_cpk(mean, std)

        results.append({
            "file": f,
            "values": vals,
            "mean": mean,
            "std": std,
            "cp": cp,
            "cpk": cpk
        })

    if not results:
        print("No magnet data found")
        return

    # Sort results by extracted date if possible (so trend is chronological)
    def sort_key(r: dict):
        d = extract_date_anywhere(r["file"])
        return d if d else datetime.max

    results.sort(key=sort_key)

    name = f"Rexair_Magnet_Timing_CpK_Report_{datetime.now():%Y-%m}.pdf"
    out = versioned_filename(REPORT_FOLDER_MAGNET, name)

    create_magnet_pdf(out, results)

    print("Magnet report created:", out)
    open_file(out)


# ============================================================
# GUI + THREADING
# ============================================================

def start_threaded(funcs: list) -> None:
    root = tk.Tk()
    root.title("Processing")
    root.geometry("420x140")
    ttk.Label(root, text="Processing...", font=("Segoe UI", 11, "bold")).pack(pady=10)
    bar = ttk.Progressbar(root, mode="indeterminate")
    bar.pack(fill="x", padx=20)
    bar.start()

    q = queue.Queue()

    def worker():
        try:
            for f in funcs:
                f(q, {})
            q.put(("done", None))
        except Exception as e:
            q.put(("error", e))

    def check():
        try:
            msg, payload = q.get_nowait()
            if msg == "done":
                bar.stop()
                root.destroy()
                return
            if msg == "error":
                bar.stop()
                root.destroy()
                messagebox.showerror("Error", f"An error occurred:\n\n{payload}")
                return
        except queue.Empty:
            pass
        root.after(200, check)

    root.after(200, check)
    threading.Thread(target=worker, daemon=True).start()
    root.mainloop()


def launch_gui() -> None:
    root = tk.Tk()
    root.title("Motor & Magnet Analysis")
    root.geometry("520x300")

    ttk.Label(root, text="Motor & Magnet Analysis",
              font=("Segoe UI", 14, "bold")).pack(pady=15)

    ttk.Button(
        root,
        text="Run Motor Analysis",
        width=44,
        command=lambda: start_threaded([run_motor_analysis])
    ).pack(pady=8)

    ttk.Button(
        root,
        text="Run Magnet Analysis",
        width=44,
        command=lambda: start_threaded([run_magnet_analysis])
    ).pack(pady=8)

    ttk.Button(
        root,
        text="Run Both",
        width=44,
        command=lambda: start_threaded([run_motor_analysis, run_magnet_analysis])
    ).pack(pady=8)

    root.mainloop()


if __name__ == "__main__":
    launch_gui()