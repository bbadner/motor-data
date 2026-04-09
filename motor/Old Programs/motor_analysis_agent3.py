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
import calendar
import subprocess
import platform

REPORT_FOLDER_MOTOR = "Motor Reports"
REPORT_FOLDER_MAGNET = "Magnet Reports"
LOGO_FILE = "rexair_logo.png"  # This file must be in the same directory

# ---------- Utility: Open File ----------
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

# ---------- Data Extract Helpers ----------
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
            if col is not None:
                values = pd.to_numeric(df.iloc[10:, col], errors="coerce").dropna()
                if len(values) > 0:
                    return values.tolist()
        return []
    except Exception as e:
        print(f"[ERROR] Reading Excel: {e}")
        return []

def extract_date_from_filename(fname):
    m = re.search(r"IN(\d{6})", fname)
    if m:
        try:
            return datetime.strptime(m.group(1), "%y%m%d")
        except:
            return None
    return None

# ---------- PDF: Motor Report ----------
def create_motor_pdf(path, months, all_values, means, stds):
    pdf = FPDF()
    pdf.add_page()

    if os.path.exists(LOGO_FILE):
        pdf.image(LOGO_FILE, x=10, y=8, w=30)
    pdf.set_font("Arial", "B", 16)
    pdf.set_xy(45, 10)
    pdf.cell(0, 10, "Rexair Motor Test Trend Analysis", ln=True)

    # Boxplot
    plt.figure(figsize=(8, 4))
    plt.boxplot(all_values, labels=[m.strftime("%b %Y") for m in months], patch_artist=True)
    plt.ylim(700, 1100)
    plt.title("Input Power (W): Monthly Distribution")
    plt.ylabel("Watts")
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig("motor_boxplot_temp.png")
    plt.close()
    pdf.image("motor_boxplot_temp.png", x=10, y=30, w=190)

    # Std Dev chart
    plt.figure(figsize=(8, 2.8))
    plt.plot([m.strftime("%b %Y") for m in months], stds, marker='o', linestyle='-')
    plt.title("Standard Deviation Over Time")
    plt.ylabel("Std Dev")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("motor_std_temp.png")
    plt.close()
    pdf.image("motor_std_temp.png", x=10, y=115, w=190)

    # Page 2: Summary table
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Monthly Summary Table", ln=True)
    pdf.set_font("Arial", "", 11)
    for i, m in enumerate(months):
        pdf.cell(0, 8, f"{m.strftime('%Y-%m')}: Mean = {means[i]:.2f} W, Std Dev = {stds[i]:.4f}", ln=True)

    pdf.output(path)
    open_file(path)
    print(f"[INFO] ✅ Motor report created: {path}")

# ---------- PDF: Magnet Report ----------
def create_magnet_pdf(path, rows):
    pdf = FPDF()
    pdf.add_page()

    if os.path.exists(LOGO_FILE):
        pdf.image(LOGO_FILE, x=10, y=8, w=30)
    pdf.set_font("Arial", "B", 16)
    pdf.set_xy(45, 10)
    pdf.cell(0, 10, "Rexair Magnet Timing CpK Report", ln=True)

    # CpK chart
    months = [calendar.month_abbr[r[0].month] for r in rows]
    cpks = [r[4] for r in rows]
    plt.figure(figsize=(8, 4))
    plt.plot(months, cpks, marker='o', linestyle='-')
    plt.axhline(y=1.33, color='red', linestyle='-', linewidth=1.5, label="CpK Threshold = 1.33")
    plt.ylim(0, 4)
    plt.title("Magnet CpK Over Time")
    plt.xlabel("Month")
    plt.ylabel("CpK")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("magnet_chart_temp.png")
    plt.close()
    pdf.image("magnet_chart_temp.png", x=10, y=30, w=190)

    # Summary table
    pdf.set_xy(10, 115)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Month Summary Table", ln=True)
    pdf.set_font("Arial", "", 11)
    for r in rows:
        d = r[0].strftime("%B %Y")
        pdf.cell(0, 8, f"{d} | Mean: {r[1]:.4f} | Std: {r[2]:.4f} | Cp: {r[3]:.2f} | CpK: {r[4]:.2f}", ln=True)

    pdf.output(path)
    open_file(path)
    print(f"[INFO] ✅ Magnet report created: {path}")

# ---------- Analysis Logic ----------
def run_motor_analysis(q=None, cancel_flag=None):
    base = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
    report_dir = os.path.join(base, REPORT_FOLDER_MOTOR)
    os.makedirs(report_dir, exist_ok=True)

    files = [f for f in os.listdir(base) if f.lower().endswith(".xlsx")]
    dated = [(extract_date_from_filename(f), f) for f in files]
    dated = [(d, f) for d, f in dated if d]
    monthly = {}

    for d, f in sorted(dated):
        m = d.strftime("%Y-%m")
        values = read_input_power(os.path.join(base, f))
        if values:
            monthly.setdefault(m, []).extend(values)

    if not monthly:
        print("❌ No Input Power data.")
        return

    all_months = sorted([datetime.strptime(k, "%Y-%m") for k in monthly])
    recent_months = all_months[-12:]
    all_values = [monthly[m.strftime("%Y-%m")] for m in recent_months]
    means = [statistics.mean(v) for v in all_values]
    stds = [statistics.stdev(v) for v in all_values]

    filename = f"Rexair_Motor_Test_Trend_Analysis_{datetime.now():%Y-%m}.pdf"
    path = os.path.join(report_dir, filename)
    create_motor_pdf(path, recent_months, all_values, means, stds)

def run_magnet_analysis(q=None, cancel_flag=None):
    base = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
    report_dir = os.path.join(base, REPORT_FOLDER_MAGNET)
    os.makedirs(report_dir, exist_ok=True)

    pdfs = [f for f in os.listdir(base) if f.lower().endswith(".pdf") and "magnet" in f.lower()]
    results = []

    for f in pdfs:
        path = os.path.join(base, f)
        with fitz.open(path) as doc:
            text = "".join(p.get_text() for p in doc)

        matches = re.findall(r"3\.\d{2,4}(?=°)", text)
        values = []
        for v in matches:
            try:
                values.append(float(v.strip("°")))
            except:
                continue

        if not values:
            print(f"[SKIP] No values from: {f}")
            continue

        mean = round(statistics.mean(values), 4)
        std = round(statistics.stdev(values), 4)
        cp = (4.2 - 3.2) / (6 * std)
        cpk = min((mean - 3.2) / (3 * std), (4.2 - mean) / (3 * std))

        date_match = re.search(r"(\d{8})", f)
        if date_match:
            dt = datetime.strptime(date_match.group(1), "%Y%m%d").date()
            results.append([dt, mean, std, cp, cpk])

    if not results:
        print("❌ No CpK data.")
        return

    filename = f"Rexair_Magnet_Timing_CpK_Report_{datetime.now():%Y-%m}.pdf"
    path = os.path.join(report_dir, filename)
    create_magnet_pdf(path, results)

# ---------- GUI ----------
def start_threaded(funcs):
    root = tk.Tk()
    root.title("Running...")
    root.geometry("400x100")
    tk.Label(root, text="Processing...", font=("Segoe UI", 12)).pack(pady=10)
    bar = ttk.Progressbar(root, mode="indeterminate")
    bar.pack(pady=5)
    bar.start()

    q = queue.Queue()
    cancel_flag = {"stop": False}
    def worker(): [f(q, cancel_flag) for f in funcs]; q.put("done")
    def check_done():
        try:
            if q.get_nowait() == "done": root.destroy()
        except queue.Empty:
            root.after(100, check_done)
    threading.Thread(target=worker, daemon=True).start()
    check_done()
    root.mainloop()

def main():
    root = tk.Tk()
    root.title("Rexair Analysis Tool")
    root.geometry("400x220")
    tk.Label(root, text="Motor & Magnet Analysis", font=("Segoe UI", 14, "bold")).pack(pady=10)
    tk.Button(root, text="Run Motor Analysis", width=40,
              command=lambda: [root.destroy(), start_threaded([run_motor_analysis])]).pack(pady=5)
    tk.Button(root, text="Run Magnet Analysis", width=40,
              command=lambda: [root.destroy(), start_threaded([run_magnet_analysis])]).pack(pady=5)
    tk.Button(root, text="Run Both Analyses", width=40,
              command=lambda: [root.destroy(), start_threaded([run_motor_analysis, run_magnet_analysis])]).pack(pady=10)
    tk.Button(root, text="Exit", width=20, command=root.destroy).pack(pady=5)
    root.mainloop()

if __name__ == "__main__":
    main()

