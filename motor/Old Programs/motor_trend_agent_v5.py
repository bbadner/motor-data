import os
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from fpdf import FPDF
import tkinter as tk
from tkinter import messagebox
import traceback
import subprocess

# --------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------
REPORT_FOLDER = "Motor Reports"
LOGO_FILE = "rexair_logo.png"
WATTS_YMIN = 700
WATTS_YMAX = 1100

# --------------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------------
def safe_popup(title, message):
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(title, message)
        root.destroy()
    except Exception:
        print(f"[Popup suppressed] {title}: {message}")

def safe_error(title, message):
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(title, message)
        root.destroy()
    except Exception:
        print(f"[Error popup suppressed] {title}: {message}")

def extract_date_from_name(filename):
    m = re.search(r"IN(\d{6})", filename)
    if m:
        try:
            return datetime.strptime(m.group(1), "%y%m%d")
        except Exception:
            return None
    return None

def find_input_power_column(df):
    for col in df.columns:
        col_lower = str(col).strip().lower()
        if "input" in col_lower and "power" in col_lower:
            return col
    return None

def read_input_power(file_path):
    try:
        # Try reading xlsx/xls automatically
        df = pd.read_excel(file_path, sheet_name=None, header=None)
        for sheet_name, sheet_data in df.items():
            sheet_data.columns = sheet_data.iloc[0]
            sheet_data = sheet_data[1:]
            col = find_input_power_column(sheet_data)
            if col:
                vals = pd.to_numeric(sheet_data[col], errors="coerce").dropna()
                if len(vals) > 0:
                    return vals
        return pd.Series(dtype=float)
    except Exception as e:
        print(f"⚠ Error reading {file_path}: {e}")
        return pd.Series(dtype=float)

# --------------------------------------------------------------------------------
# PDF Report
# --------------------------------------------------------------------------------
def create_pdf_report(output_path, logo_path, avg_data, std_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    if os.path.exists(logo_path):
        pdf.image(logo_path, 10, 8, 33)
    pdf.cell(80)
    pdf.cell(30, 10, "Rexair Motor Performance Report", ln=1, align="C")

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=1)

    # Charts
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    plt.subplots_adjust(hspace=0.4)

    # Box-and-whisker for average watts
    avg_data.boxplot(ax=ax1)
    ax1.set_title("Distribution of Input Power (W) by Month")
    ax1.set_ylabel("Watts")
    ax1.set_ylim(WATTS_YMIN, WATTS_YMAX)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Line for std dev
    std_data.plot(ax=ax2, marker="o")
    ax2.set_title("Monthly Standard Deviation of Input Power (W)")
    ax2.set_ylabel("Std Dev (W)")
    ax2.set_xlabel("Month")
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Save chart image
    chart_img = "chart_temp.png"
    plt.tight_layout()
    plt.savefig(chart_img, bbox_inches="tight")
    plt.close(fig)

    pdf.image(chart_img, x=10, y=50, w=190)
    os.remove(chart_img)

    pdf.output(output_path, "F")

# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------
def main():
    try:
        base_dir = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
        report_dir = os.path.join(base_dir, REPORT_FOLDER)
        os.makedirs(report_dir, exist_ok=True)

        excel_files = [f for f in os.listdir(base_dir) if f.lower().endswith((".xlsx", ".xls"))]
        if not excel_files:
            safe_error("No Files Found", "No Excel files (.xlsx/.xls) found in this folder.")
            return

        today = datetime.now()
        cutoff = today - timedelta(days=365)

        monthly_data = {}
        for fname in excel_files:
            fdate = extract_date_from_name(fname)
            if not fdate or fdate < cutoff:
                continue
            fpath = os.path.join(base_dir, fname)
            watts = read_input_power(fpath)
            if len(watts) == 0:
                continue
            month_label = fdate.strftime("%Y-%m")
            monthly_data.setdefault(month_label, []).extend(watts.tolist())

        if not monthly_data:
            safe_error("No Valid Data", "No valid 'Input Power' data found in the past 12 months.")
            return

        avg_series = pd.Series({m: pd.Series(v).mean() for m, v in monthly_data.items()})
        std_series = pd.Series({m: pd.Series(v).std() for m, v in monthly_data.items()})
        avg_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in monthly_data.items()]))

        report_name = f"Rexair_Motor_Performance_Report_{today.strftime('%Y-%m-%d')}.pdf"
        report_path = os.path.join(report_dir, report_name)

        create_pdf_report(report_path, os.path.join(base_dir, LOGO_FILE), avg_df, std_series)

        safe_popup("Report Complete", f"Report saved to:\n{report_path}")
        subprocess.run(f'explorer "{report_dir}"')
    except Exception as e:
        with open("error_log.txt", "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        safe_error("Error", f"An error occurred:\n{e}")

# --------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
