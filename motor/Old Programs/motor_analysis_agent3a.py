# =================================================
# PART 1  IMPORTS, CONSTANTS AND UTILITY FUNCTIONS
# =================================================

import os
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from fpdf import FPDF
import tkinter as tk
from tkinter import ttk, messagebox, filedialog  # ✅ Add 'filedialog' here
import threading
import statistics
import fitz  # PyMuPDF
import queue
import calendar
import subprocess
import platform
import glob
import webbrowser
from PyPDF2 import PdfMerger

REPORT_FOLDER_MOTOR = "Motor Reports"
REPORT_FOLDER_MAGNET = "Magnet Reports"
LOGO_FILE = "rexair_logo.png"  # Logo must be in the same directory
motor_output_pdf = "Rexair_Motor_Test_Trend_Analysis.pdf"
magnet_output_pdf = "Rexair_Magnet_Timing_CpK_Report.pdf"
combined_output_pdf = "Rexair_Combined_QA_Report.pdf"

# =================================================
# PART 2 FILE HANDLING AND DATE PARSING
# =================================================

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

# Use C652 format for motor date (not IN format)
def extract_date_from_filename(fname):
    m = re.search(r"C\d{3}(\d{6})", fname)  # Match C6521231116 pattern
    if m:
        try:
            return datetime.strptime(m.group(1), "%y%m%d")
        except:
            return None
    return None

# =================================================
# PART 3 REPORT GENERATIONS
# Adds your logo
# Displays a textual summary for the most recent month
# Charts (boxplot + standard deviation) saved into PDF
# Highlights high or low SD ranges with action suggestions
# =================================================

class PDF(FPDF):
    def header(self):
        if os.path.exists(LOGO_FILE):
            self.image(LOGO_FILE, 10, 8, 33)
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Rexair Motor Test Trend Analysis", ln=True, align="C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def create_motor_pdf(path, recent_months, all_values, means, stds):
    pdf = PDF()
    pdf.add_page()

    # Summary section
    try:
        most_recent = max(recent_months)
        recent_data = all_values[recent_months.index(most_recent)]
        mean = means[recent_months.index(most_recent)]
        std = stds[recent_months.index(most_recent)]

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "📊 Summary of Most Recent Month", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 8, f"Month: {most_recent.strftime('%B %Y')}")
        pdf.multi_cell(0, 8, f"Mean Input Power: {mean:.2f} W")
        pdf.multi_cell(0, 8, f"Standard Deviation: {std:.2f} W")

        if std > 25:
            pdf.set_text_color(220, 50, 50)
            pdf.multi_cell(0, 8, "⚠️ High variability detected. Investigate sources of process variation.")
            pdf.set_text_color(0, 0, 0)
        elif std < 10:
            pdf.multi_cell(0, 8, "✅ Stable process with low variation.")
        else:
            pdf.multi_cell(0, 8, "ℹ️ Moderate variation. Monitor closely.")
        pdf.ln(5)
    except Exception as e:
        print(f"[WARN] Unable to write summary: {e}")

    # Boxplot of all data
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.boxplot(all_values, labels=[m.strftime("%b %Y") for m in recent_months], patch_artist=True)
        ax.set_title("Input Power (W): Monthly Distribution")
        ax.set_ylabel("Watts")
        ax.set_ylim(700, 1100)
        plt.xticks(rotation=45)
        chart_path = os.path.join(path, "motor_boxplot.png")
        fig.tight_layout()
        fig.savefig(chart_path)
        plt.close(fig)
        pdf.image(chart_path, x=10, y=None, w=190)
    except Exception as e:
        print(f"[ERROR] Creating boxplot: {e}")

    # Standard Deviation Chart
    try:
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.plot([m.strftime("%b %Y") for m in recent_months], stds, marker="o")
        ax.set_title("Standard Deviation of Input Power")
        ax.set_ylabel("SD (Watts)")
        ax.set_ylim(0, max(30, max(stds) + 5))
        plt.xticks(rotation=45)
        chart2_path = os.path.join(path, "motor_sdplot.png")
        fig.tight_layout()
        fig.savefig(chart2_path)
        plt.close(fig)
        pdf.image(chart2_path, x=10, y=None, w=190)
    except Exception as e:
        print(f"[ERROR] Creating SD chart: {e}")

    report_path = os.path.join(path, f"{motor_output_pdf[:-4]}_{datetime.now().strftime('%Y-%m')}.pdf")
    pdf.output(report_path)
    print(f"[INFO] Motor report created: {report_path}")
    return report_path

# =================================================
# PART 4 MOTOR DATA PROCESSING AND DATE PARSING LOGIC
# Extracts correct motor date from filenames using C652-based serial (e.g. C6521231116-0001 → Nov 16, 2023).
# Parses and aggregates motor input power data month by month.
# Limits output to last 12 months.
# Handles bad/missing files gracefully.
# Sends everything to the report builder from Part 3.
# =================================================

def parse_date_from_filename(filename):
    # Extract the 13-digit serial code that starts with "C652" (e.g., C6521231116-0001)
    match = re.search(r"(C652\d{9})", filename)
    if match:
        date_str = match.group(1)[4:]  # Skip 'C652'
        try:
            return datetime.strptime(date_str, "%y%m%d")  # yy mm dd format (23 11 16)
        except ValueError:
            return None
    return None

def run_motor_analysis(path, motor_files):
    print("[INFO] Starting motor data analysis...")

    data_by_month = defaultdict(list)
    all_values = []
    for file in motor_files:
        date = parse_date_from_filename(file)
        if not date:
            continue

        try:
            df = pd.read_excel(file)
        except Exception as e:
            print(f"[WARN] Could not read {file}: {e}")
            continue

        # Attempt to find the correct power column
        power_column = None
        for col in df.columns:
            if isinstance(col, str) and "power" in col.lower():
                power_column = col
                break

        if power_column is None:
            print(f"[WARN] Skipping file with no power column: {file}")
            continue

        values = pd.to_numeric(df[power_column], errors='coerce').dropna().tolist()
        if not values:
            print(f"[WARN] No valid power data in: {file}")
            continue

        month_key = datetime(date.year, date.month, 1)
        data_by_month[month_key].extend(values)
        all_values.extend(values)

    if not data_by_month:
        print("[ERROR] No valid motor data found.")
        return

    # Sort months and keep only last 12
    sorted_months = sorted(data_by_month.keys())[-12:]
    monthly_values = [data_by_month[m] for m in sorted_months]
    monthly_means = [np.mean(vals) for vals in monthly_values]
    monthly_stds = [np.std(vals) for vals in monthly_values]

    # Generate the PDF report
    report_path = create_motor_pdf(path, sorted_months, monthly_values, monthly_means, monthly_stds)
    return report_path

# =================================================
# PART 5
# Uses a GUI folder picker (like before).
# Collects motor Excel files (*.xlsx, *.xls) based on "motor" in filename.
# Calls the full motor analysis + report pipeline.
# Opens the final report automatically when complete (as requested).
# =================================================

if __name__ == "__main__":
    print("📊 Rexair Motor Data Analysis Tool")

    # Automatically use the folder where the script is located
    selected_folder = os.getcwd()
    print(f"🔍 Using current directory for analysis: {selected_folder}")

    # Step 1: Identify relevant motor and magnet files in the folder
    motor_files = glob.glob(os.path.join(selected_folder, '*motor*.csv'))
    magnet_files = glob.glob(os.path.join(selected_folder, '*magnet*.csv'))

    if not motor_files:
        print("⚠️ No motor data files found in the selected folder.")
        exit()

    if not magnet_files:
        print("⚠️ No magnet data files found in the selected folder.")
        exit()

    # Step 2: Run the analysis using those files
    all_values, means, stds, recent_months = run_motor_analysis(motor_files, magnet_files)

    # Step 3: Create output file names using timestamp
    timestamp = datetime.now().strftime("%Y-%m")
    motor_output_path = f"Rexair_Motor_Test_Trend_Analysis_{timestamp}.pdf"
    summary_template = "Motor_Analysis_Summary_Template.pdf"

    # Step 4: Generate dynamic summary addendum
    summary_output_path = f"Motor_Analysis_Summary_{timestamp}.pdf"
    create_summary_addendum(summary_output_path, stds)

    # Step 5: Merge analysis and summary into one PDF
    final_report_path = f"Rexair_Motor_Report_WITH_SUMMARY_{timestamp}.pdf"
    merger = PdfMerger()
    if os.path.exists(motor_output_path):
        merger.append(motor_output_path)
    if os.path.exists(summary_output_path):
        merger.append(summary_output_path)
    merger.write(final_report_path)
    merger.close()

    print(f"✅ Full report generated: {final_report_path}")
