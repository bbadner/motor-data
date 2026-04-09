import os
import sys
import re
import statistics
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from fpdf import FPDF
import matplotlib.pyplot as plt
import fitz  # PyMuPDF
import pandas as pd
from datetime import datetime

# ---------------------------------------------------
# Utility Functions
# ---------------------------------------------------
def resource_path(relative_path):
    """Get absolute path for PyInstaller exe compatibility"""
    if getattr(sys, "frozen", False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def calculate_cpk(values, target=3.7, tol=0.5):
    """Compute Cp, CpK using sample stdev and correct min(Cpu, Cpl)"""
    if len(values) < 2:
        return 0, 0, 0
    m = statistics.mean(values)
    s = statistics.stdev(values)
    usl, lsl = target + tol, target - tol
    if s == 0:
        return m, 0, 0
    cp = (usl - lsl) / (6 * s)
    cpu = (usl - m) / (3 * s)
    cpl = (m - lsl) / (3 * s)
    cpk = min(cpu, cpl)
    return m, cp, cpk

# ---------------------------------------------------
# Tkinter Progress Window
# ---------------------------------------------------
def create_progress_window(title):
    win = tk.Toplevel()
    win.title(title)
    win.geometry("400x150")
    tk.Label(win, text=title, font=("Arial", 12)).pack(pady=10)
    progress = ttk.Progressbar(win, orient="horizontal", mode="determinate", length=300)
    progress.pack(pady=10)
    cancel_flag = tk.BooleanVar(value=False)
    def cancel(): cancel_flag.set(True)
    tk.Button(win, text="Cancel", command=cancel).pack(pady=10)
    win.protocol("WM_DELETE_WINDOW", cancel)
    return win, progress, cancel_flag

# ---------------------------------------------------
# Motor Data Analysis
# ---------------------------------------------------
def run_motor_analysis(progress=None, cancel_flag=None):
    base_dir = os.path.dirname(sys.executable if getattr(sys, "frozen", False) else __file__)
    data_files = [f for f in os.listdir(base_dir) if f.lower().endswith(".xlsx")]
    total_files = len(data_files)
    if progress: progress["maximum"] = total_files
    if not data_files:
        print("[INFO] ❌ No valid motor data found.")
        return

    report_folder = os.path.join(base_dir, "Motor Reports")
    ensure_folder(report_folder)
    report_path = os.path.join(report_folder, f"Rexair_Motor_Test_Trend_Analysis_{datetime.now():%Y-%m}_v3.pdf")

    all_data = []
    for idx, file in enumerate(data_files, start=1):
        if cancel_flag and cancel_flag.get(): return
        file_path = os.path.join(base_dir, file)
        try:
            df = pd.read_excel(file_path, sheet_name=None)
            for _, sheet in df.items():
                sheet.columns = sheet.columns.astype(str)
                mask = sheet.apply(lambda row: row.astype(str).str.contains("High Speed", case=False).any(), axis=1)
                if mask.any():
                    watts = pd.to_numeric(sheet.iloc[:, 7:15].stack(), errors="coerce").dropna()
                    if not watts.empty:
                        date_match = re.search(r"C6521(\d{6})", file)
                        date = date_match.group(1) if date_match else "000000"
                        all_data.append({
                            "Month": date[:6],
                            "Average": watts.mean(),
                            "StdDev": watts.std()
                        })
        except Exception as e:
            print(f"[DEBUG] Skipped {file}: {e}")
        if progress: progress["value"] = idx

    if not all_data:
        print("[INFO] ❌ No valid motor data found.")
        return

    df_summary = pd.DataFrame(all_data)
    df_summary["Month"] = pd.to_datetime(df_summary["Month"], format="%y%m%d").dt.to_period("M")
    df_summary = df_summary.groupby("Month").agg({"Average": "mean", "StdDev": "mean"}).reset_index()

    # --- PDF Creation ---
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Rexair Motor Test Trend Analysis", ln=True, align="C")
    pdf.ln(10)

    # --- Box Plot ---
    plt.figure(figsize=(10, 5))
    plt.boxplot(df_summary["Average"], patch_artist=True)
    plt.title("Monthly Average Wattage (Box Plot)")
    plt.ylabel("Average Watts")
    plt.ylim(700, 1100)
    plt.grid(True, linestyle="--", alpha=0.6)
    box_img = os.path.join(base_dir, "motor_boxplot.png")
    plt.savefig(box_img)
    plt.close()

    # --- Std Dev Line Plot ---
    plt.figure(figsize=(10, 4))
    plt.plot(df_summary["Month"].astype(str), df_summary["StdDev"], marker="o", linestyle="-")
    plt.title("Monthly Standard Deviation")
    plt.ylabel("Std Dev")
    plt.grid(True, linestyle="--", alpha=0.6)
    std_img = os.path.join(base_dir, "motor_std.png")
    plt.savefig(std_img)
    plt.close()

    pdf.image(box_img, x=10, w=180)
    pdf.image(std_img, x=10, y=130, w=180)
    pdf.add_page()
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, "Summary Table", ln=True)
    pdf.ln(5)
    for _, row in df_summary.iterrows():
        pdf.cell(0, 8, f"{row['Month']}: Avg = {row['Average']:.2f}, StdDev = {row['StdDev']:.3f}", ln=True)

    pdf.output(report_path)
    print(f"[INFO] ✅ Motor report created: {report_path}")

# ---------------------------------------------------
# Magnet Timing CpK Analysis
# ---------------------------------------------------
def run_magnet_analysis(progress=None, cancel_flag=None):
    base_dir = os.path.dirname(sys.executable if getattr(sys, "frozen", False) else __file__)
    pdf_files = [f for f in os.listdir(base_dir) if f.lower().endswith(".pdf")]
    total_files = len(pdf_files)
    if progress: progress["maximum"] = total_files
    if not pdf_files:
        print("[INFO] ❌ No magnet timing data found.")
        return

    results = []
    for idx, pdf_file in enumerate(pdf_files, start=1):
        if cancel_flag and cancel_flag.get(): return
        pdf_path = os.path.join(base_dir, pdf_file)
        try:
            doc = fitz.open(pdf_path)
            text = "".join(page.get_text("text") for page in doc)
            matches = re.findall(r"3\.\d{2}", text)
            values = [float(v) for v in matches if 3.0 <= float(v) <= 4.5]

            if values:
                m, cp, cpk = calculate_cpk(values)
                date_match = re.search(r"(\d{6,8})", pdf_file)
                month = date_match.group(1) if date_match else "000000"
                results.append({
                    "Month": month,
                    "Mean": m,
                    "Cp": cp,
                    "CpK": cpk,
                    "StDev": statistics.stdev(values)
                })
        except Exception as e:
            print(f"[DEBUG] Skipped {pdf_file}: {e}")
        if progress: progress["value"] = idx

    if not results:
        print("[INFO] ❌ No valid CpK results computed.")
        return

    def safe_parse_date(x):
        x = str(x)
        if len(x) == 6: return pd.to_datetime(x + "01", format="%Y%m%d")
        elif len(x) == 8: return pd.to_datetime(x, format="%Y%m%d")
        else: return pd.NaT

    df = pd.DataFrame(results)
    df["Month"] = df["Month"].apply(safe_parse_date).dt.to_period("M")
    df = df.groupby("Month").mean().reset_index()

    report_folder = os.path.join(base_dir, "Magnet Reports")
    ensure_folder(report_folder)
    report_path = os.path.join(report_folder, f"Rexair_Magnet_Timing_CpK_Report_{datetime.now():%Y-%m}.pdf")

    # --- CpK Chart ---
    plt.figure(figsize=(10, 5))
    plt.plot(df["Month"].astype(str), df["CpK"], marker="o", linestyle="-", label="CpK")
    plt.axhline(1.33, color="red", linestyle="--", label="Target CpK = 1.33")
    plt.title("Magnet Timing CpK Trend")
    plt.xlabel("Month")
    plt.ylabel("CpK Value")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    chart_path = os.path.join(base_dir, "magnet_cpk.png")
    plt.savefig(chart_path)
    plt.close()

    # --- PDF Report ---
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Rexair Magnet Timing CpK Analysis", ln=True, align="C")
    pdf.image(chart_path, x=15, w=180)
    pdf.ln(100)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, "Monthly Summary (3.7 ± 0.5°):", ln=True)
    pdf.ln(5)
    for _, row in df.iterrows():
        pdf.cell(0, 8, f"{row['Month']} | Cp={row['Cp']:.3f} | CpK={row['CpK']:.3f} | StDev={row['StDev']:.3f}", ln=True)

    pdf.output(report_path)
    print(f"[INFO] ✅ Magnet report created: {report_path}")

# ---------------------------------------------------
# GUI Menu
# ---------------------------------------------------
def start_threaded_analysis(funcs):
    win, progress, cancel_flag = create_progress_window("Running Analysis...")
    thread = threading.Thread(target=lambda: [f(progress, cancel_flag) for f in funcs])
    thread.start()
    def check_thread():
        if thread.is_alive():
            win.after(100, check_thread)
        else:
            win.destroy()
    check_thread()

def main_menu():
    root = tk.Tk()
    root.title("Rexair Analysis Menu")
    tk.Label(root, text="Select analysis option:", font=("Arial", 12)).pack(pady=10)
    tk.Button(root, text="A – Motor Test Trend Analysis",
              command=lambda:[root.destroy(), start_threaded_analysis([run_motor_analysis])],
              width=40).pack(pady=5)
    tk.Button(root, text="B – Magnet Timing CpK Analysis",
              command=lambda:[root.destroy(), start_threaded_analysis([run_magnet_analysis])],
              width=40).pack(pady=5)
    tk.Button(root, text="C – Run Both",
              command=lambda:[root.destroy(), start_threaded_analysis([run_motor_analysis, run_magnet_analysis])],
              width=40).pack(pady=5)
    tk.Button(root, text="Cancel", command=root.destroy, width=40).pack(pady=10)
    root.mainloop()

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if __name__ == "__main__":
    main_menu()

