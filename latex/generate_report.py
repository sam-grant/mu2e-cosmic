#!/usr/bin/env python3
"""
LaTeX Report Generator with Introduction

This script generates a LaTeX report from analysis results.
It reads CSV tables and includes PNG plots for each analysis configuration.
Now includes comprehensive introduction section with datasets, beam timing, and cutsets.
"""

import os
import pandas as pd
from pathlib import Path
import argparse
import shutil


def convert_png_to_pdf(png_path, pdf_path):
    """
    Convert PNG to PDF using PIL/Pillow while preserving original size and DPI.
    
    Args:
        png_path (str): Path to input PNG file
        pdf_path (str): Path to output PDF file
    
    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        from PIL import Image
        
        # Open PNG image
        with Image.open(png_path) as img:
            # Get original DPI if available, otherwise use a reasonable default
            original_dpi = img.info.get('dpi')
            if original_dpi is None:
                # If no DPI info, assume 96 DPI (common screen resolution)
                # or calculate based on typical plot sizes
                dpi = 96
            else:
                # DPI can be a tuple (x_dpi, y_dpi) or a single value
                dpi = original_dpi[0] if isinstance(original_dpi, tuple) else original_dpi
            
            print(f"    Original PNG: {img.size[0]}x{img.size[1]} pixels, DPI: {dpi}")
            
            # Convert to RGB if necessary (PDFs don't support transparency)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save as PDF preserving the original DPI
            # This ensures the PDF has the same physical dimensions as the PNG
            img.save(pdf_path, 'PDF', 
                    dpi=(dpi, dpi),  # Explicitly set both x and y DPI
                    quality=95,
                    optimize=True)
        
        print(f"    Converted PNG to PDF: {os.path.basename(pdf_path)} (preserved DPI: {dpi})")
        return True
    
    except ImportError:
        print("    Warning: PIL/Pillow not available. Install with: pip install pillow")
        return False
    except Exception as e:
        print(f"    Warning: PNG to PDF conversion failed: {e}")
        return False

def fix_math_formatting(text):
    """
    Fix LaTeX math formatting in text - preserve $ in math expressions.
    
    Args:
        text (str): Input text that might contain math
    
    Returns:
        str: Text with proper math formatting
    """
    text = str(text)
    text = text.replace('_', '\\_')
    text = text.replace('%', '\\%')
    text = text.replace('&', '\\&')
    
    # Don't escape $ if it's part of a math expression (contains \text{)
    if '$' in text and '\\text{' in text:
        # This is a math expression, leave $ alone
        pass
    # else:
    #     # Regular text, escape $
    #     text = text.replace('$', '\\$')
    
    return text


def csv_to_latex_table(csv_path, caption="", label="", output_dir="", use_adjustbox=False):
    """
    Convert CSV file to LaTeX table format and copy CSV to tables directory.
    
    Args:
        csv_path (str): Path to the CSV file
        caption (str): Table caption
        label (str): Table label for referencing
        output_dir (str): Directory to copy the CSV file to
        use_adjustbox (bool): Whether to wrap in adjustbox for centering
    
    Returns:
        str: LaTeX table code
    """
    if not os.path.exists(csv_path):
        return "% Table not found: " + csv_path + "\n"
    
    try:
        df = pd.read_csv(csv_path)
        # df = df.round(3) # round to 3 decimal places
        
        # Copy CSV file to tables directory
        if output_dir:
            tables_dir = os.path.join(output_dir, "tables")
            os.makedirs(tables_dir, exist_ok=True)
            csv_filename = os.path.basename(csv_path)
            dest_csv_path = os.path.join(tables_dir, csv_filename)
            shutil.copy2(csv_path, dest_csv_path)
            print(f"  Copied table: {csv_filename}")
        
        # Generate custom LaTeX table
        num_cols = len(df.columns)
        
        # Create column specification - first column left-aligned, rest centered
        if num_cols > 1:
            col_spec = 'l|' + 'c' * (num_cols - 1)
        else:
            col_spec = 'l'
        
        # Start building the table
        if use_adjustbox:
            latex_table = "\\begin{table}[H]\n"
            latex_table += "\\centering{}\n"
            latex_table += "\\adjustbox{width=1.5\\textwidth,center}{\n"
            latex_table += "\\begin{tabular}{" + col_spec + "}\n"
        else:
            latex_table = "\\begin{table}[H]\n"
            latex_table += "\\centering{}\n"
            latex_table += "\\begin{tabular}{" + col_spec + "}\n"
        
        latex_table += "\\hline\n"
        latex_table += "\\hline\n"
        
        # Add header row with proper math formatting
        header_cols = []
        for col in df.columns:
            header_cols.append(fix_math_formatting(col))
        
        header_row = ' & '.join(header_cols)
        latex_table += header_row + " \\\\\n"
        latex_table += "\\hline\n"
        
        # Add data rows
        for _, row in df.iterrows():
            row_data = []
            for val in row:
                if pd.isna(val):
                    row_data.append('NaN')
                elif isinstance(val, (int, float)):
                    # Format numbers nicely
                    if isinstance(val, float):
                        if val == 0.0:
                            row_data.append('0')
                        elif abs(val) >= 1000000:
                            row_data.append(f'{val:.0f}')
                        elif abs(val) >= 1000:
                            row_data.append(f'{val:.0f}')
                        elif abs(val) >= 1:
                            row_data.append(f'{val:.2f}')
                        else:
                            row_data.append(f'{val:.3f}')
                    else:
                        row_data.append(str(val))
                else:
                    # Apply math formatting to text
                    row_data.append(fix_math_formatting(val))
            
            row_str = ' & '.join(row_data)
            latex_table += row_str + " \\\\\n"
        
        # Close the table
        latex_table += "\\hline\n"
        latex_table += "\\hline\n"
        latex_table += "\\end{tabular}\n"
        
        if use_adjustbox:
            latex_table += "}\n"  # Close adjustbox
        
        latex_table += "\\caption{" + caption + "}\n"
        latex_table += "\\label{" + label + "}\n"
        latex_table += "\\end{table}\n\n"
        
        return latex_table
    
    except Exception as e:
        return "% Error reading " + csv_path + ": " + str(e) + "\n"


def generate_introduction_section():
    """
    Generate the introduction section with datasets, beam timing, and cutsets information.
    
    Returns:
        str: LaTeX content for the introduction section
    """
    
    intro_content = ""
    
    # Introduction header
    intro_content += "\\section{Introduction}\n\n"
    
    intro_content += "This genetated report presents the results of cosmic-ray-induced background analysis for the Mu2e experiment."
    
    # Datasets subsection
    intro_content += "\\subsection{Datasets}\n\n"
    
    intro_content += "The analysis utilises Monte Carlo simulation datasets."
    
    # Create a LaTeX table for the dataset information
    intro_content += "\\begin{table}[H]\n"
    intro_content += "\\centering\n"
    # intro_content += "\\footnotesize\n"
    intro_content += "\\begin{tabular}{l|ccc}\n"
    intro_content += "\\hline\n"
    intro_content += "\\hline\n"
    intro_content += "\\textbf{Type} & \\textbf{Livetime [days]} & \\textbf{Triggered events} & \\textbf{Generated events} \\\\\n"
    intro_content += "\\hline\n"
    intro_content += "CosmicCRY & 124 & 10,050,055 & 41.1B \\\\\n"
    intro_content += "CosmicCRY Mix2BB & 124 & 10,050,055 & 41.1B \\\\\n"
    intro_content += "CeEndpoint & -- & 2,166,583 & 4M \\\\\n"
    intro_content += "CeEndpoint Mix2BB & -- & 2,166,583 & 4M \\\\\n"
    intro_content += "\\hline\n"
    intro_content += "\\hline\n"
    intro_content += "\\end{tabular}\n"
    intro_content += "\\caption{Summary of Monte Carlo datasets.}\n"
    intro_content += "\\label{tab:datasets}\n"
    intro_content += "\\end{table}\n\n"
    
    # Beam timing structure subsection
    intro_content += "\\subsection{Beam timing structure}\n\n"
    
    intro_content += "The Mu2e experiment operates with a pulsed beam structure. The beam structure is characterised "
    intro_content += "by supercycles with distinct on-spill and off-spill periods.\n\n"
    
    # Beam timing table
    intro_content += "\\begin{table}[H]\n"
    intro_content += "\\centering\n"
    intro_content += "\\begin{tabular}{l|cc}\n"
    intro_content += "\\hline\n"
    intro_content += "\\hline\n"
    intro_content += "\\textbf{Parameter} & \\textbf{Single batch} & \\textbf{Two batch} \\\\\n"
    intro_content += "\\hline\n"
    intro_content += "Supercycle [ms] & 1333 & 1400 \\\\\n"
    intro_content += "Number of spills & 4 & 8 \\\\\n"
    intro_content += "Spill duration [ms] & 107.3 & 43.1 \\\\\n"
    intro_content += "Total spill time [ms] & 429.2 & 344.8 \\\\\n"
    intro_content += "\\hdashline\n"
    intro_content += "On-spill fraction & 0.322 (32.2\\%) & 0.246 (24.6\\%) \\\\\n"
    intro_content += "Off-spill fraction & 0.678 (67.8\\%) & 0.754 (75.4\\%) \\\\\n"
    intro_content += "\\hline\n"
    intro_content += "\\hline\n"
    intro_content += "\\end{tabular}\n"
    intro_content += "\\caption{Beam timing structure parameters for single and two-batch modes.}\n"
    intro_content += "\\label{tab:beam_timing}\n"
    intro_content += "\\end{table}\n\n"
    
    # Cutsets subsection
    intro_content += "\\subsection{Cutsets}\n\n"
    
    intro_content += "The analyses employ hierarchical cutsets based on the SU2020 analysis."
    intro_content += "The cutsets are designed to select high-quality downstream electron tracks "
    intro_content += "originating from the stopping target\\footnote{See arXiv:2210.11380 for "
    intro_content += "details.}.\n\n"
    
    # Preselection cuts
    intro_content += "\\subsubsection{Preselection}\n\n"
    intro_content += "All analysis configurations apply a common set of preselection cuts:\n\n"
    intro_content += "\\begin{itemize}\n"
    intro_content += "\\item \\texttt{has\\_trk\\_front}: Track intersects the tracker entrance\n"
    intro_content += "\\item \\texttt{is\\_reco\\_electron}: Track fit uses electron hypothesis\n"
    intro_content += "\\item \\texttt{one\\_reco\\_electron}: One electron fit per event\n"
    intro_content += "\\item \\texttt{is\\_downstream}: Downstream tracks ($p_z > 0$ at tracker entrance)\n"
    intro_content += "\\item \\texttt{is\\_truth\\_electron}: Truth PID (track parents are electrons)\n"
    intro_content += "\\end{itemize}\n\n"

    intro_content += "Note that \\texttt{is\\_downstream} is defined through all tracker planes (front, middle, and back).\n\n"
    
    # Signal-like background cuts
    intro_content += "\\subsubsection{Signal-like background cuts}\n\n"
    intro_content += "Common signal-like background selection cuts include:\n\n"
    intro_content += "\\begin{itemize}\n"
    intro_content += "\\item \\texttt{unvetoed}: No veto requirement ($|\\Delta t| \\geq 150$ ns)\n"
    intro_content += "\\item \\texttt{within\\_ext\\_win}: Extended momentum window (100 < $p$ < 110 MeV/c)\n"
    intro_content += "\\item \\texttt{within\\_sig\\_win}: Signal momentum window (103.6 < $p$ < 104.9 MeV/c)\n"
    intro_content += "\\end{itemize}\n\n"
    
    # Signal-like cuts variations
    intro_content += "\\subsubsection{Signal-like cut variations}\n\n"
    intro_content += "These analyses may include combinations of track quality (\\texttt{trkqual}) and "
    intro_content += "time uncertainty (\\texttt{t0err}) cuts:\n\n"
    
    intro_content += "\\begin{table}[H]\n"
    intro_content += "\\centering\n"
    intro_content += "\\begin{tabular}{l|cc}\n"
    intro_content += "\\hline\n"
    intro_content += "\\hline\n"
    intro_content += "\\textbf{trkqual $\\backslash$ t0err} & \\textbf{< 0.9 ns} & \\textbf{> 0 (none)} \\\\\n"
    intro_content += "\\hline\n"
    intro_content += "> 0 (none) & SU2020a & SU2020d \\\\\n"
    intro_content += "> 0.2 & SU2020b & SU2020e \\\\\n"
    intro_content += "> 0.8 & SU2020c & SU2020f \\\\\n"
    intro_content += "\\hline\n"
    intro_content += "\\hline\n"
    intro_content += "\\end{tabular}\n"
    intro_content += "\\caption{Matrix of cutset variations.}\n"
    intro_content += "\\label{tab:cutset_matrix}\n"
    intro_content += "\\end{table}\n\n"
    
    # Detailed cut descriptions
    intro_content += "Each cutset variation includes the following track-based cuts:\n\n"
    intro_content += "\\begin{itemize}\n"
    intro_content += "\\item \\texttt{good\\_trkqual}: Track quality requirement (varies by cutset)\n"
    intro_content += "\\item \\texttt{within\\_t0}: Track time at tracker entrance (640 < $t_0$ < 1650 ns)\n"
    intro_content += "\\item \\texttt{within\\_t0err}: Track fit time uncertainty (varies by cutset)\n"
    intro_content += "\\item \\texttt{has\\_hits}: Minimum number of active tracker hits (> 20)\n"
    intro_content += "\\item \\texttt{within\\_d0}: Distance of closest approach ($d_0 < 100$ mm)\n"
    intro_content += "\\item \\texttt{within\\_pitch\\_angle\\_lo}: Lower pitch angle cut ($p_z/p_t > 0.5$)\n"
    intro_content += "\\item \\texttt{within\\_pitch\\_angle\\_hi}: Upper pitch angle cut ($p_z/p_t < 1.0$)\n"
    intro_content += "\\item \\texttt{within\\_lhr\\_max\\_hi}: Loop helix maximum radius ($R_{\\text{max}} < 680$ mm)\n"
    intro_content += "\\end{itemize}\n\n"
    
    intro_content += "The SU2020b cutset is used as the primary reference configuration, since it is the most "
    intro_content += "to the baseline SU2020 analysis. Note that \\texttt{within\\_t0} and \\texttt{within\\_t0err} "
    intro_content += "are measured at the tracker middle, since that is where $t_{0}$ is defined, while extrapolated "
    intro_content +=  "parameters are measured at the tracker front.\n\n"
    
    # # Analysis methodology
    # intro_content += "\\subsection{Analysis methods}\n\n"
    
    # intro_content += "The analysis framework processes each dataset through the defined cutset configurations. "
    # intro_content += "Key results include:\n\n"
    
    # intro_content += "\\begin{itemize}\n"
    # intro_content += "\\item \\textbf{Cut flow Analysis}: Sequential application of cuts to determine selection efficiency\n"
    # intro_content += "\\item \\textbf{Momentum Window Studies}: Optimization of signal and extended momentum windows\n"
    # intro_content += "\\item \\textbf{Background Characterization}: Detailed study of cosmic ray background distributions\n"
    # intro_content += "\\item \\textbf{Signal Extraction}: Statistical methods for signal identification and quantification\n"
    # intro_content += "\\end{itemize}\n\n"
    
    # intro_content += "Each analysis configuration is identified by a label combining the cutset type, dataset "
    # intro_content += "configuration, and timing window (e.g., \\texttt{SU2020b\\_CRY\\_onspill-LH\\_au}). This "
    # intro_content += "systematic approach enables comprehensive comparison across different analysis strategies.\n\n"
    
    intro_content += "\\newpage\n\n"
    
    return intro_content


def generate_latex_report(ana_labels, output_file="report"):
    """
    Generate a complete LaTeX report for multiple analysis configurations.
    Creates a self-contained directory with all images and tables copied locally.
    
    Args:
        ana_labels (list): List of analysis label strings
        output_file (str): Output directory name
    """
    
    # Remove .tex extension if provided
    if output_file.endswith('.tex'):
        output_file = output_file[:-4]
    
    # Create main report directory
    report_dir = "../output/latex/" + output_file
    os.makedirs(report_dir, exist_ok=True)
    
    # Create subdirectories
    images_dir = os.path.join(report_dir, "images")
    tables_dir = os.path.join(report_dir, "tables")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    
    print("Creating report in: " + report_dir)
    
    # Full path for output file
    full_output_path = os.path.join(report_dir, output_file + ".tex")
    
    # Start building LaTeX content
    latex_content = ""
    
    # Document header
    latex_content += "\\documentclass[11pt,a4paper]{article}\n"
    latex_content += "\\usepackage[utf8]{inputenc}\n"
    latex_content += "\\usepackage[T1]{fontenc}\n"
    latex_content += "\\usepackage{geometry}\n"
    latex_content += "\\usepackage[final]{graphicx}\n"
    latex_content += "\\usepackage{booktabs}\n"
    latex_content += "\\usepackage{longtable}\n"
    latex_content += "\\usepackage{array}\n"
    latex_content += "\\usepackage{float}\n"
    latex_content += "\\usepackage{caption}\n"
    latex_content += "\\usepackage{subcaption}\n"
    latex_content += "\\usepackage{amsmath}\n"
    latex_content += "\\usepackage{amsfonts}\n"
    latex_content += "\\usepackage{amssymb}\n"
    latex_content += "\\usepackage{hyperref}\n"
    latex_content += "\\usepackage{pdflscape}\n"
    latex_content += "\\usepackage{adjustbox}\n"
    latex_content += "\\usepackage{tabularx}\n"
    latex_content += "\\usepackage{enumitem}\n"
    latex_content += "\\usepackage{arydshln}\n"
    latex_content += "\n"
    latex_content += "\\geometry{margin=2.5cm}\n"
    latex_content += "\\graphicspath{{images/}}\n"
    latex_content += "\\DeclareGraphicsExtensions{.pdf,.png,.jpg}\n"
    latex_content += "\n"
    latex_content += "% Custom commands for better formatting\n"
    latex_content += "\\newcommand{\\MuE}{Mu2e}\n"
    latex_content += "\\newcommand{\\trkqual}{\\texttt{trkqual}}\n"
    latex_content += "\\newcommand{\\toerr}{\\texttt{t0err}}\n"
    latex_content += "\n"
    latex_content += "\\title{Cosmic-ray-induced background analysis report\\\\}\n"
    latex_content += "\\author{Samuel Grant}\n"
    latex_content += "\\date{\\today}\n"
    latex_content += "\n"
    latex_content += "\\begin{document}\n"
    latex_content += "\n"
    latex_content += "\\maketitle\n"
    latex_content += "\n"
    latex_content += "\\begin{abstract}\n"
    latex_content += "This is an auto-generated report presenting the results of cosmic-ray-induced background analysis for the Mu2e experiment.\n"
    latex_content += "\\end{abstract}\n"
    latex_content += "\n"
    latex_content += "\\tableofcontents\n"
    latex_content += "\\newpage\n"
    latex_content += "\n"
    
    # Add the introduction section
    latex_content += generate_introduction_section()

    # Generate content for each analysis label
    for ana_label in ana_labels:
        print("Processing analysis: " + ana_label)
        
        # Paths
        results_path = "../output/results/" + ana_label
        source_images_path = "../output/images/" + ana_label
        
        # Create subdirectory for this analysis in images
        ana_images_dir = os.path.join(images_dir, ana_label)
        os.makedirs(ana_images_dir, exist_ok=True)
        
        # Create title page for this analysis
        safe_ana_label = ana_label.replace('_', '\\_')
        latex_content += "\\begin{titlepage}\n"
        latex_content += "    \\centering\n"
        latex_content += "    \\vspace*{\\fill}\n"
        latex_content += "    \n"
        latex_content += "    \\Large\n"
        latex_content += "    Analysis results for configuration\n"
        latex_content += "    \n"
        latex_content += "    \\vspace{2cm}\n"
        latex_content += "    \n"
        latex_content += "    \\Huge\n"
        latex_content += "    \\texttt{" + safe_ana_label + "}\n"
        latex_content += "    \n"
        latex_content += "    \\vspace*{\\fill}\n"
        latex_content += "    \n"
        latex_content += "    \\large\n"
        latex_content += "    Generated on \\today\n"
        latex_content += "    \n"
        latex_content += "    \\vspace{0.5cm}\n"
        latex_content += "    \n"
        latex_content += "    Samuel Grant\n"
        latex_content += "    \n"
        latex_content += "\\end{titlepage}\n"
        latex_content += "\\newpage\n\n"
        
        # Section header
        section_title = ana_label.replace('_', '\\_').replace('-', '--')
        latex_content += "\\section{" + section_title + "}\n\n"
        
        # Check if results directory exists
        if not os.path.exists(results_path):
            latex_content += "\\textbf{Warning:} Results directory not found: \\texttt{" + results_path.replace('_', '\\_') + "}\n\n"
            continue
        
        latex_content += "This section presents the analysis results for configuration \\texttt{" + safe_ana_label + "}. "
        
        # Add plots first
        latex_content += "\\subsection{Plots}\n\n"
        
        # Check if source images directory exists
        if os.path.exists(source_images_path):
            # Define source paths for images
            plot1_source = os.path.join(source_images_path, "h1o_1x3_mom_windows.png")
            plot2_source = os.path.join(source_images_path, "h1o_3x3_summary.png")
            
            # Process first plot
            plot1_exists = False
            plot1_file = None
            
            if os.path.exists(plot1_source):
                # Try to convert PNG to PDF for better quality
                plot1_pdf_dest = os.path.join(ana_images_dir, "h1o_1x3_mom_windows.pdf")
                if convert_png_to_pdf(plot1_source, plot1_pdf_dest):
                    plot1_file = ana_label + "/h1o_1x3_mom_windows.pdf"
                    plot1_exists = True
                    print("  Using PDF: " + ana_label + "/h1o_1x3_mom_windows.pdf")
                else:
                    # Fallback to PNG
                    plot1_png_dest = os.path.join(ana_images_dir, "h1o_1x3_mom_windows.png")
                    shutil.copy2(plot1_source, plot1_png_dest)
                    plot1_file = ana_label + "/h1o_1x3_mom_windows.png"
                    plot1_exists = True
                    print("  Using PNG: " + ana_label + "/h1o_1x3_mom_windows.png")
            else:
                print("  Warning: Plot 1 not found at " + plot1_source)

            # # Process first plot - use high-res PNG directly
            # if os.path.exists(plot1_source):
            #     plot1_dest = os.path.join(ana_images_dir, "h1o_1x3_mom_windows.png")
            #     shutil.copy2(plot1_source, plot1_dest)
            #     plot1_file = ana_label + "/h1o_1x3_mom_windows.png"
            #     plot1_exists = True
            #     print(f"  Using high-res PNG: {ana_label}/h1o_1x3_mom_windows.png")

    
            # Process second plot
            plot2_exists = False
            plot2_file = None

            # # Process second plot - use high-res PNG directly
            # if os.path.exists(plot2_source):
            #     plot2_dest = os.path.join(ana_images_dir, "h1o_3x3_summary.png")
            #     shutil.copy2(plot2_source, plot2_dest)
            #     plot2_file = ana_label + "/h1o_3x3_summary.png"
            #     plot2_exists = True
            #     print(f"  Using high-res PNG: {ana_label}/h1o_3x3_summary.png")


            
            if os.path.exists(plot2_source):
                # Try to convert PNG to PDF for better quality
                plot2_pdf_dest = os.path.join(ana_images_dir, "h1o_3x3_summary.pdf")
                if convert_png_to_pdf(plot2_source, plot2_pdf_dest):
                    plot2_file = ana_label + "/h1o_3x3_summary.pdf"
                    plot2_exists = True
                    print("  Using PDF: " + ana_label + "/h1o_3x3_summary.pdf")
                else:
                    # Fallback to PNG
                    plot2_png_dest = os.path.join(ana_images_dir, "h1o_3x3_summary.png")
                    shutil.copy2(plot2_source, plot2_png_dest)
                    plot2_file = ana_label + "/h1o_3x3_summary.png"
                    plot2_exists = True
                    print("  Using PNG: " + ana_label + "/h1o_3x3_summary.png")
            else:
                print("  Warning: Plot 2 not found at " + plot2_source)
            
            # Generate LaTeX for plots
            clean_ana_label = ana_label.replace('-', '_').replace('_', '')
            
            if plot1_exists:
                latex_content += "\\begin{figure}[H]\n"
                latex_content += "    \\centering\n"
                latex_content += "    \\includegraphics[width=\\textwidth]{" + plot1_file + "}\n"
                latex_content += "    \\caption{Momentum windows analysis for configuration " + safe_ana_label + ". "
                latex_content += "Momentum distributions over three windows: wide, extended, and signal.}\n"
                latex_content += "    \\label{fig:h1o1x3" + clean_ana_label + "}\n"
                latex_content += "\\end{figure}\n\n"
            
            if plot2_exists:
                latex_content += "\\begin{figure}[H]\n"
                latex_content += "    \\centering\n"
                latex_content += "    \\includegraphics[width=\\textwidth]{" + plot2_file + "}\n"
                latex_content += "    \\caption{Summary of cut parameters for configuration " + safe_ana_label + ".}"
                latex_content += "    \\label{fig:h1o3x3" + clean_ana_label + "}\n"
                latex_content += "\\end{figure}\n\n"

            # if plot1_exists:
            #     latex_content += "\\begin{figure}[H]\n"
            #     latex_content += "    \\centering\n"
            #     # Use width=\\textwidth or specify exact dimensions
            #     latex_content += "    \\includegraphics[width=\\textwidth,keepaspectratio]{" + plot1_file + "}\n"
            #     # Alternative: specify exact size
            #     # latex_content += "    \\includegraphics[width=16cm,height=12cm,keepaspectratio]{" + plot1_file + "}\n"
            #     latex_content += "    \\caption{Momentum windows analysis for configuration " + safe_ana_label + ". "
            #     latex_content += "Momentum distributions over three windows: wide, extended, and signal.}\n"
            #     latex_content += "    \\label{fig:h1o1x3" + clean_ana_label + "}\n"
            #     latex_content += "\\end{figure}\n\n"
            
            # if plot2_exists:
            #     latex_content += "\\begin{figure}[H]\n"
            #     latex_content += "    \\centering\n"
            #     # For larger plots, you might want to use a specific size
            #     latex_content += "    \\includegraphics[width=\\textwidth,keepaspectratio]{" + plot2_file + "}\n"
            #     # Alternative for very detailed plots:
            #     # latex_content += "    \\includegraphics[width=1.2\\textwidth,keepaspectratio]{" + plot2_file + "}\n"
            #     latex_content += "    \\caption{Summary of cut parameters for configuration " + safe_ana_label + ".}"
            #     latex_content += "    \\label{fig:h1o3x3" + clean_ana_label + "}\n"
            #     latex_content += "\\end{figure}\n\n"
    
        else:
            latex_content += "\\textbf{Warning:} Images directory not found: \\texttt{" + source_images_path.replace('_', '\\_') + "}\n\n"
            print("  Warning: Images directory not found: " + source_images_path)
        
        # Start landscape for tables
        latex_content += "\\begin{landscape}\n\n"
        
        # Add cut flow table
        latex_content += "\\subsection{Cut flow}\n\n"
        latex_content += "The cut flow table for configuration " + safe_ana_label + ".\n\n"
        
        cut_flow_path = os.path.join(results_path, "cut_flow.csv")
        latex_content += csv_to_latex_table(
            cut_flow_path,
            caption="Cut flow table for configuration " + safe_ana_label + ".",
            label="tab:cutflow_" + clean_ana_label,
            output_dir=report_dir,
            use_adjustbox=False
        )

        latex_content += "\\newpage\n\n"
        
        # Add analysis table
        latex_content += "\\subsection{Analysis results}\n\n"
        latex_content += "Analysis results for configuration " + safe_ana_label + ".\n\n"
        
        analysis_path = os.path.join(results_path, "analysis.csv")
        
        if os.path.exists(analysis_path):
            latex_content += csv_to_latex_table(
                analysis_path,
                caption="Analysis results for configuration " + safe_ana_label + ".",
                use_adjustbox=True
            )
        else:
            latex_content += "\\textbf{Warning:} Analysis results file not found: \\texttt{" + analysis_path.replace('_', '\\_') + "}\n\n"
            print("  Warning: Analysis results file not found: " + analysis_path)
        
        # End landscape and add page break
        latex_content += "\\end{landscape}\n"
        latex_content += "\\newpage\n\n"
    
    # Add bibliography section
    # latex_content += "\\section{References}\n\n"
    # latex_content += "\\begin{thebibliography}{9}\n\n"
    # latex_content += "\\bibitem{SU2020}\n"
    # latex_content += "Mu2e Collaboration,\n"
    # latex_content += "\\textit{Search for muon-to-electron conversion: an experiment in the Stopped Muon (SU2020) approach},\n"
    # latex_content += "arXiv:2210.11380 [hep-ex] (2022).\n\n"
    # latex_content += "\\bibitem{MDC2020}\n"
    # latex_content += "Mu2e Collaboration,\n"
    # latex_content += "\\textit{Mu2e Data Challenge 2020 Monte Carlo Production},\n"
    # latex_content += "Mu2e-doc-33162 (2020).\n\n"
    # latex_content += "\\bibitem{CRY}\n"
    # latex_content += "C. Hagmann et al.,\n"
    # latex_content += "\\textit{Cosmic-ray shower generator (CRY) for Monte Carlo transport codes},\n"
    # latex_content += "IEEE Nuclear Science Symposium Conference Record, 1143-1146 (2007).\n\n"
    # latex_content += "\\end{thebibliography}\n\n"
    
    # Document footer
    latex_content += "\\end{document}\n"
    
    # Write to file
    with open(full_output_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    print("LaTeX report generated: " + full_output_path)
    return full_output_path


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Generate LaTeX report from analysis results')
    parser.add_argument('ana_labels', nargs='+', help='Analysis labels to include in report')
    parser.add_argument('-o', '--output', default='report', help='Output directory name (default: report)')
    parser.add_argument('--compile', action='store_true', help='Compile LaTeX to PDF using pdflatex')
    
    args = parser.parse_args()
    
    # Generate the report
    latex_file = generate_latex_report(args.ana_labels, args.output)
    
    # Optionally compile to PDF
    if args.compile:
        try:
            import subprocess
            print("Compiling LaTeX to PDF...")
            # Change to the report directory for compilation
            latex_dir = os.path.dirname(latex_file)
            latex_filename = os.path.basename(latex_file)
            
            result = subprocess.run(['pdflatex', latex_filename], 
                                  capture_output=True, text=True, cwd=latex_dir)
            if result.returncode == 0:
                pdf_path = os.path.join(latex_dir, latex_filename.replace('.tex', '.pdf'))
                print("PDF generated successfully: " + pdf_path)
                # Run twice for proper references
                subprocess.run(['pdflatex', latex_filename], 
                             capture_output=True, text=True, cwd=latex_dir)
                print("Second pass completed for proper references")
            else:
                print("LaTeX compilation failed:\n" + result.stderr)
        except FileNotFoundError:
            print("pdflatex not found. Please install LaTeX to compile PDFs.")


if __name__ == "__main__":
    # Example usage if run directly
    if len(os.sys.argv) == 1:
        print("Example usage:")
        print("python generate_report.py config1 config2 config3")
        print("python generate_report.py config1 config2 -o my_report --compile")
        print("\nOr use directly in code:")
        
        # Example with sample configurations
        example_configs = ["analysis_v1", "analysis_v2", "baseline"]
        generate_latex_report(example_configs, "example_report")
    else:
        main()