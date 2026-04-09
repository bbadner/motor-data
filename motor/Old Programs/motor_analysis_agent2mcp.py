# ---------------------------------------------------
# Section 1: Imports, Logging, and Helper Functions
# ---------------------------------------------------

import os
import re
import sys
import fitz  # PyMuPDF for PDF parsing
import threading
import statistics
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from fpdf import FPDF
import tkinter as tk
from tkinter import ttk, messagebox
import queue
import subprocess

# ---------- LOGGING SETUP ----------
def init_logger():
    base_dir = os.path.dirname(sys.executable if getattr(sys, "frozen", False) else __file__)
    log_path = os.path.join(base_dir, "log.txt")
    return open(log_path, "a", encoding="utf-8")

LOG_FILE = init_logger()

def log(message):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    full_message = f"{timestamp} {message}"
    print(full_message)
    LOG_FILE.write(full_message + "\n")
    LOG_FILE.flush()

def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        log(f"Created folder: {path}")

def open_file(path):
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform.startswith("darwin"):
            subprocess.call(["open", path])
        else:
            subprocess.call(["xdg-open", path])
        log(f"Opened file: {path}")
    except Exception as e:
        log(f"[WARN] Could not open file: {e}")

# ---------------------------------------------------
# Section 2: Progress Window and Cancel Logic
# ---------------------------------------------------

def create_progress_window(title="Processing"):
    win = tk.Toplevel()
    win.title(title)
    win.geometry("400x150")
    win.resizable(False, False)

    label = tk.Label(win, text="Processing data... Please wait.", font=("Segoe UI", 10))
    label.pack(pady=10)

    progress = ttk.Progressbar(win, length=320, mode="determinate")
    progress.pack(pady=10)

    cancel_flag = {"stop": False}

    def cancel():
        cancel_flag["stop"] = True
        messagebox.showinfo("Cancelled", "Operation cancelled by user.")
        win.destroy()
        log("Operation cancelled by user.")

    cancel_button = tk.Button(win, text="Cancel", command=cancel)
    cancel_button.pack(pady=5)

    return win, progress, cancel_flag

# ---------------------------------------------------
# Section 3: Motor Test Trend Analysis
# ---------------------------------------------------

def run_motor_analysis(q=None, cancel_flag=None):
    log("[INFO] Starting Rexair Motor Test Trend Analysis...")

    base_path = os.path.dirname(sys.executable if getattr(sys, "frozen", False) else __file__)
    excel_files = [f for f in os.listdir(base_path) if f.lower().endswith((".xlsx", ".xls"))]

    if not excel_files:
        log("[INFO] ❌ No Excel motor data files found in this directory.")
        return

    all_data = []

    for f in excel_files:
        try:
            file_path = os.path.join(base_path, f)
            xl = pd.ExcelFile(file_path)

            found = False
            for sheet_name in xl.sheet_names:
                df = xl.parse(sheet_name)
                if "High Speed" in " ".join(df.columns.astype(str)):
                    found = True
                    break
                if df.astype(str).apply(lambda x: x.str.contains("High Speed", case=False, na=False)).any().any():
                    found = True
                    break
            if not found:
                log(f"[Motor] Skipped {f}: No 'High Speed (Open)' data located.")
                continue

            # ✅ FIX: More robust Input Power column detection
            power_col = None
            for col in df.columns:
                col_str = str(col).lower().replace(" ", "").replace("_", "")
                if "inputpower" in col_str:
                    power_col = col
                    break
            if power_col is None:
                log(f"[Motor] Skipped {f}: No 'Input Power(W)' column found.")
                continue

            data_series = pd.to_numeric(df[power_col], errors="coerce").dropna()
            if len(data_series) < 20:
                log(f"[Motor] Skipped {f}: Not enough valid Input Power data ({len(data_series)} pts).")
                continue

            avg = data_series.mean()
            std = data_series.std()

            match = re.search(r"C6521(\\d{6})", f)
            if match:
                date = datetime.strptime(match.group(1), "%y%m%d").date()
            else:
                log(f"[Motor] Skipped {f}: No valid date found in filename.")
                continue

            all_data.append([f, avg, std, date])
            log(f"[Motor] {f}: Avg={avg:.2f} W, Std={std:.2f}, Date={date}")

        except Exception as e:
            log(f"[Motor ERROR] {f}: {e}")

    if not all_data:
        log("[INFO] ❌ No valid motor trend data compiled.")
        return

    df = pd.DataFrame(all_data, columns=["File", "Average", "Stdev", "Date"])
    df["Month"] = df["Date"].apply(lambda x: x.strftime("%Y-%m"))

    grouped = df.groupby("Month").agg({"Average": "mean", "Stdev": "mean"}).reset_index()
    grouped = grouped.sort_values("Month")
    if len(grouped) > 12:
        grouped = grouped.tail(12)
        log("[INFO] Applied rolling 12-month filter to motor trend data.")

    os.makedirs("Motor Reports", exist_ok=True)
    pdf_path = os.path.join("Motor Reports",
        f"Rexair_Motor_Test_Trend_Analysis_{datetime.now():%Y-%m}.pdf")

    plt.figure(figsize=(8, 5))
    plt.boxplot(df["Average"], vert=True, patch_artist=True)
    plt.ylim(700, 1100)
    plt.title("Motor Input Power (High Speed Open) – Box Plot")
    plt.ylabel("Watts")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("motor_box_temp.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(grouped["Month"], grouped["Stdev"], marker="o")
    plt.title("Motor Input Power – Standard Deviation Trend")
    plt.xlabel("Month")
    plt.ylabel("Standard Deviation (W)")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("motor_stdev_temp.png", bbox_inches="tight")
    plt.close()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Rexair Motor Test Trend Analysis", ln=True, align="C")
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 8, "Report includes the most recent 12 months of available data.", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Report Date: {datetime.now():%B %d, %Y}", ln=True)

    pdf.image("motor_box_temp.png", x=15, y=40, w=180)
    pdf.image("motor_stdev_temp.png", x=15, y=120, w=180)

    pdf.set_xy(10, 200)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Monthly Summary:", ln=True)
    pdf.set_font("Arial", "", 11)
    for _, row in grouped.iterrows():
        pdf.cell(0, 8,
                 f"{row['Month']} | Avg={row['Average']:.1f} W | Stdev={row['Stdev']:.2f}",
                 ln=True)

    pdf.output(pdf_path)
    log(f"[INFO] ✅ Motor report created: {pdf_path}")
    try:
        os.startfile(pdf_path)
    except Exception:
        log("[INFO] ⚠️ Could not auto-open motor PDF.")

# ---------------------------------------------------
# Section 4: Magnet Timing CpK Analysis
# ---------------------------------------------------

MAGNET_TARGET = 3.7
MAGNET_TOL = 0.5
LSL = MAGNET_TARGET - MAGNET_TOL
USL = MAGNET_TARGET + MAGNET_TOL

def run_magnet_analysis(q=None, cancel_flag=None):
    log("[INFO] Starting Rexair Magnet Timing CpK Analysis...")

    base_path = os.path.dirname(sys.executable if getattr(sys, "frozen", False) else __file__)
    pdf_files = [f for f in os.listdir(base_path) if f.lower().endswith(".pdf") and "magnet" in f.lower()]

    if not pdf_files:
        log("[INFO] ❌ No magnet timing data found.")
        return

    results = []
    for f in pdf_files:
        try:
            path = os.path.join(base_path, f)
            with fitz.open(path) as doc:
                text = "".join(page.get_text("text") for page in doc)

            text = re.sub(r"3\\.(\\d)\\s+(\\d)", r"3.\\1\\2", text)
            values = [float(v) for v in re.findall(r"3\\.\\d{2,4}", text) if 3.0 < float(v) < 4.5]

            if not values:
                log(f"[Magnet] ⚠️ No valid 3.7±0.5° values found in {f}")
                continue

            # ✅ FIX: Use population standard deviation
            mean = round(statistics.mean(values), 4)
            std = round(statistics.pstdev(values), 4)

            cp = (USL - LSL) / (6 * std)
            cpk_lower = (mean - LSL) / (3 * std)
            cpk_upper = (USL - mean) / (3 * std)
            cpk = min(cpk_lower, cpk_upper)

            match = re.search(r"(\\d{8})", f)
            if match:
                date = datetime.strptime(match.group(1), "%Y%m%d").date()
                results.append([date, mean, std, cp, cpk])
                log(f"[Magnet] {f}: mean={mean:.3f}, std={std:.3f}, Cp={cp:.3f}, CpK={cpk:.3f}, values={len(values)}")

        except Exception as e:
            log(f"[Magnet ERROR] {f}: {e}")

    if not results:
        log("[INFO] ❌ No valid CpK results computed.")
        return

    df = pd.DataFrame(results, columns=["Date", "Mean", "Stdev", "Cp", "CpK"])
    df["Month"] = df["Date"].apply(lambda x: x.strftime("%Y-%m"))
    df = df.sort_values("Date")

    os.makedirs("Magnet Reports", exist_ok=True)
    pdf_path = os.path.join("Magnet Reports", f"Rexair_Magnet_Timing_CpK_Report_{datetime.now():%Y-%m}.pdf")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Rexair Magnet Timing CpK Analysis", ln=True, align="C")

    plt.figure(figsize=(8, 5))
    plt.plot(df["Month"], df["CpK"], marker="o")
    plt.axhline(1.33, color="red", linestyle="--", label="Minimum Acceptable CpK")
    plt.title("Magnet Timing CpK Trend (12-Month Rolling)")
    plt.ylabel("CpK Value")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("magnet_chart_temp.png", bbox_inches="tight")
    plt.close()

    pdf.image("magnet_chart_temp.png", x=20, y=40, w=170)
    pdf.set_xy(10, 130)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Monthly Summary (3.7 ± 0.5°):", ln=True)
    pdf.set_font("Arial", "", 11)

    for _, row in df.iterrows():
        pdf.cell(0, 8,
                 f"{row['Month']} | Cp={row['Cp']:.3f} | CpK={row['CpK']:.3f} | "
                 f"StDev={row['Stdev']:.4f} | Mean={row['Mean']:.3f}",
                 ln=True)

    pdf.output(pdf_path)
    log(f"[INFO] ✅ Magnet report created: {pdf_path}")
    try:
        os.startfile(pdf_path)
    except Exception:
        log("[INFO] ⚠️ Could not auto-open PDF file.")

# ---------------------------------------------------
# Remaining Sections (Threaded Execution and Main GUI)
# ---------------------------------------------------

# These parts are unchanged; you can keep them as in your original script.



# ---------------------------------------------------
# Section 5: Threaded Analysis Runner
# ---------------------------------------------------

def start_threaded_analysis(funcs):
    """Runs selected analyses in a worker thread while updating progress bar."""
    win, progress, cancel_flag = create_progress_window("Rexair Analysis Progress")
    q = queue.Queue()

    def worker():
        """Background worker thread to run each selected function."""
        for func in funcs:
            log(f"Starting function: {func.__name__}")
            try:
                func(q, cancel_flag)
            except Exception as e:
                log(f"[ERROR] Exception in {func.__name__}: {e}")
        q.put(("done", None))

    def update_ui():
        """Continuously polls queue for progress updates."""
        try:
            while True:
                msg, value = q.get_nowait()
                if msg == "set_max":
                    progress["maximum"] = value
                    log(f"Progress max set to {value}")
                elif msg == "set_val":
                    progress["value"] = value
                elif msg == "done":
                    log("All selected analyses complete.")
                    win.destroy()
                    return
        except queue.Empty:
            pass
        win.after(100, update_ui)

    # Start thread
    threading.Thread(target=worker, daemon=True).start()
    update_ui()
    win.mainloop()

# ---------------------------------------------------
# Section 6: Main GUI Menu and Entry Point
# ---------------------------------------------------

def main():
    """Main program GUI menu."""
    root = tk.Tk()
    root.title("Rexair Analysis Menu")
    root.geometry("420x260")
    root.resizable(False, False)

    tk.Label(
        root,
        text="Rexair Motor & Magnet Analysis",
        font=("Segoe UI", 14, "bold"),
    ).pack(pady=15)

    tk.Label(
        root,
        text="Select an option below:",
        font=("Segoe UI", 11)
    ).pack(pady=5)

    # Buttons for options
    tk.Button(
        root,
        text="A – Motor Test Trend Analysis",
        command=lambda: [root.destroy(), start_threaded_analysis([run_motor_analysis])],
        width=40
    ).pack(pady=5)

    tk.Button(
        root,
        text="B – Magnet Timing CpK Analysis",
        command=lambda: [root.destroy(), start_threaded_analysis([run_magnet_analysis])],
        width=40
    ).pack(pady=5)

    tk.Button(
        root,
        text="C – Run Both Analyses",
        command=lambda: [root.destroy(), start_threaded_analysis([run_motor_analysis, run_magnet_analysis])],
        width=40
    ).pack(pady=10)

    # Exit button
    tk.Button(
        root,
        text="Exit",
        command=root.destroy,
        width=20
    ).pack(pady=5)

    root.mainloop()


if __name__ == "__main__":
    log("=== Rexair Analysis Agent Launched ===")
    try:
        main()
    except Exception as e:
        log(f"[FATAL] Program crashed: {e}")
    finally:
        log("=== Rexair Analysis Agent Closed ===")
        LOG_FILE.close()
