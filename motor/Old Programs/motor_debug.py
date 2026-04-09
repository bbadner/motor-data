# ✅ DEBUG LOGGING VERSION

import os
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from fpdf import FPDF
import tkinter as tk
from tkinter import ttk
import threading
import statistics
import fitz
import queue

REPORT_FOLDER_MOTOR = "Motor Reports"
REPORT_FOLDER_MAGNET = "Magnet Reports"

# === MOTOR HELPERS ===
def find_input_power_column(df):
    for i in range(min(10, len(df))):
        row = df.iloc[i].astype(str).str.lower()
        for idx, val in enumerate(row):
            if "input" in val and "power" in val:
                return idx
    return None

def read_input_power(filepath):
    try:
        excel = pd.ExcelFile(filepath)
        for sheet in excel.sheet_names:
            df = excel.parse(sheet, header=None)
            col = find_input_power_column(df)
            print(f"🧾 Checking sheet '{sheet}' in file '{os.path.basename(filepath)}': column found -> {col}")
            if col is not None:
                values = pd.to_numeric(df.iloc[10:, col], errors="coerce").dropna()
                print(f"📊 Extracted {len(values)} values from sheet '{sheet}'")
                if len(values) > 0:
                    return values.tolist()
        return []
    except Exception as e:
        print(f"[ERROR] Failed to read Excel: {e}")
        return []

def extract_date_from_filename(fname):
    m = re.search(r"IN(\d{6})", fname)
    if m:
        try:
            return datetime.strptime(m.group(1), "%y%m%d")
        except Exception:
            return None
    return None

def run_motor_analysis(q=None, cancel_flag=None):
    base_dir = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
    report_dir = os.path.join(base_dir, REPORT_FOLDER_MOTOR)
    os.makedirs(report_dir, exist_ok=True)

    excel_files = [f for f in os.listdir(base_dir) if f.lower().endswith(".xlsx")]
    print("📂 Excel files found:", excel_files)
    dated_files = []
    for f in excel_files:
        d = extract_date_from_filename(f)
        print(f"🗂️ {f} -> Parsed date: {d}")
        if d:
            dated_files.append((d, f))

    if not dated_files:
        print("❌ No valid dated Excel files found.")
        return

    dated_files.sort()
    monthly_data = {}
    for fdate, fname in dated_files:
        month = fdate.strftime("%Y-%m")
        values = read_input_power(os.path.join(base_dir, fname))
        print(f"🔍 File: {fname} | Month: {month} | Values count: {len(values)}")
        if values:
            monthly_data.setdefault(month, []).extend(values)

    if not monthly_data:
        print("❌ No valid 'Input Power' data extracted.")
        return

    for month, vals in monthly_data.items():
        print(f"✅ {month}: {len(vals)} values")

    print("✅ Motor data collection complete.")
    # You can extend with PDF generation or CpK if needed.

# === MAGNET ANALYSIS ===
def run_magnet_analysis(q=None, cancel_flag=None):
    base_dir = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
    report_dir = os.path.join(base_dir, REPORT_FOLDER_MAGNET)
    os.makedirs(report_dir, exist_ok=True)

    pdf_files = [f for f in os.listdir(base_dir) if f.lower().endswith(".pdf") and "magnet" in f.lower()]
    print("📂 Magnet PDFs found:", pdf_files)

    results = []
    for f in pdf_files:
        try:
            path = os.path.join(base_dir, f)
            with fitz.open(path) as doc:
                text = "".join(page.get_text("text") for page in doc)

            # Fix split decimals
            text = re.sub(r"3\.(\d)\s+(\d)", r"3.\1\2", text)

            # Extract float values in expected range
            values = [float(v) for v in re.findall(r"3\.\d{2,4}", text) if 3.0 < float(v) < 4.5]
            print(f"📄 {f}: Extracted values: {values[:10]} (total {len(values)})")

            if not values:
                continue

            mean = round(statistics.mean(values), 4)
            std = round(statistics.pstdev(values), 4)
            cp = (4.2 - 3.2) / (6 * std)
            cpk_lower = (mean - 3.2) / (3 * std)
            cpk_upper = (4.2 - mean) / (3 * std)
            cpk = min(cpk_lower, cpk_upper)

            print(f"📈 {f}: mean={mean}, std={std}, Cp={cp:.3f}, CpK={cpk:.3f}")

            match = re.search(r"(\d{8})", f)
            if match:
                date = datetime.strptime(match.group(1), "%Y%m%d").date()
                results.append([date, mean, std, cp, cpk])

        except Exception as e:
            print(f"[ERROR] Magnet PDF: {e}")

    if not results:
        print("❌ No valid CpK results.")
        return

    print("✅ Magnet CpK analysis complete.")

# === THREAD + UI ===
def start_threaded_analysis(funcs):
    root = tk.Tk()
    root.title("Running Analysis")
    root.geometry("400x100")
    tk.Label(root, text="Processing...", font=("Segoe UI", 12)).pack(pady=10)
    bar = ttk.Progressbar(root, mode="indeterminate")
    bar.pack(pady=5)
    bar.start()

    q = queue.Queue()
    cancel_flag = {"stop": False}

    def worker():
        for func in funcs:
            func(q, cancel_flag)
        q.put("done")

    def check_done():
        try:
            if q.get_nowait() == "done":
                root.destroy()
        except queue.Empty:
            root.after(100, check_done)

    threading.Thread(target=worker, daemon=True).start()
    check_done()
    root.mainloop()

# === MAIN MENU ===
def main():
    root = tk.Tk()
    root.title("Rexair Debug Analysis")
    root.geometry("400x220")

    tk.Label(root, text="Debug Mode – Rexair Analysis", font=("Segoe UI", 14, "bold")).pack(pady=10)
    tk.Button(root, text="Run Motor Analysis", width=40,
              command=lambda: [root.destroy(), start_threaded_analysis([run_motor_analysis])]).pack(pady=5)

    tk.Button(root, text="Run Magnet Analysis", width=40,
              command=lambda: [root.destroy(), start_threaded_analysis([run_magnet_analysis])]).pack(pady=5)

    tk.Button(root, text="Run Both Analyses", width=40,
              command=lambda: [root.destroy(), start_threaded_analysis([run_motor_analysis, run_magnet_analysis])]).pack(pady=10)

    tk.Button(root, text="Exit", width=20, command=root.destroy).pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    main()
