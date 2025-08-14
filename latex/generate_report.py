#!/usr/bin/env python3
"""
LaTeX Report Generator

This script generates a LaTeX report from analysis results.
It reads CSV tables and includes PNG plots for each analysis configuration.
"""

import os
import pandas as pd
from pathlib import Path
import argparse
import shutil


def convert_png_to_pdf(png_path, pdf_path):
    """
    Convert PNG to PDF using PIL/Pillow for better LaTeX quality.
    
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
            
            # Save as PDF with high quality
            img.save(pdf_path, 'PDF', resolution=300.0, quality=95)
        
        print(f"    Converted PNG to PDF: {os.path.basename(pdf_path)}")
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
    else:
        # Regular text, escape $
        text = text.replace('$', '\\$')
    
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
        return f"% Table not found: {csv_path}\n"
    
    try:
        df = pd.read_csv(csv_path)
        df = df.round(3)
        
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
            latex_table += f"\\begin{{tabular}}{{{col_spec}}}\n"
        else:
            latex_table = "\\begin{table}[H]\n"
            latex_table += "\\centering{}\n"
            latex_table += f"\\begin{{tabular}}{{{col_spec}}}\n"
        
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
                            row_data.append(f'{val:.6f}')
                        else:
                            row_data.append(f'{val:.6f}')
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
        
        latex_table += f"\\caption{{{caption}}}\n"
        latex_table += f"\\label{{{label}}}\n"
        latex_table += "\\end{table}\n\n"
        
        return latex_table
    
    except Exception as e:
        return f"% Error reading {csv_path}: {str(e)}\n"


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
    report_dir = f"../output/latex/{output_file}"
    os.makedirs(report_dir, exist_ok=True)
    
    # Create subdirectories
    images_dir = os.path.join(report_dir, "images")
    tables_dir = os.path.join(report_dir, "tables")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    
    print(f"Creating report in: {report_dir}")
    
    # Full path for output file
    full_output_path = os.path.join(report_dir, f"{output_file}.tex")
    
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
    latex_content += "\n"
    latex_content += "\\geometry{margin=2.5cm}\n"
    latex_content += "\\graphicspath{{images/}}\n"
    latex_content += "\\DeclareGraphicsExtensions{.pdf,.png,.jpg}\n"
    latex_content += "\n"
    latex_content += "\\title{Analysis Results Report}\n"
    latex_content += "\\author{Samuel Grant}\n"
    latex_content += "\\date{\\today}\n"
    latex_content += "\n"
    latex_content += "\\begin{document}\n"
    latex_content += "\n"
    latex_content += "\\maketitle\n"
    latex_content += "\\tableofcontents\n"
    latex_content += "\\newpage\n"
    latex_content += "\n"

    # Generate content for each analysis label
    for ana_label in ana_labels:
        print(f"Processing analysis: {ana_label}")
        
        # Paths
        results_path = f"../output/results/{ana_label}"
        source_images_path = f"../output/images/{ana_label}"
        
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
        latex_content += "    Analysis Results for Configuration\n"
        latex_content += "    \n"
        latex_content += "    \\vspace{2cm}\n"
        latex_content += "    \n"
        latex_content += "    \\Huge\n"
        latex_content += f"    \\texttt{{{safe_ana_label}}}\n"
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
        latex_content += f"\\section{{{section_title}}}\n\n"
        
        # Check if results directory exists
        if not os.path.exists(results_path):
            latex_content += f"\\textbf{{Warning:}} Results directory not found: \\texttt{{{results_path.replace('_', '\\_')}}}\n\n"
            continue
        
        latex_content += f"This section presents the analysis results for configuration \\texttt{{{safe_ana_label}}}\n\n"
        
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
                    plot1_file = f"{ana_label}/h1o_1x3_mom_windows.pdf"
                    plot1_exists = True
                    print(f"  Using PDF: {ana_label}/h1o_1x3_mom_windows.pdf")
                else:
                    # Fallback to PNG
                    plot1_png_dest = os.path.join(ana_images_dir, "h1o_1x3_mom_windows.png")
                    shutil.copy2(plot1_source, plot1_png_dest)
                    plot1_file = f"{ana_label}/h1o_1x3_mom_windows.png"
                    plot1_exists = True
                    print(f"  Using PNG: {ana_label}/h1o_1x3_mom_windows.png")
            else:
                print(f"  Warning: Plot 1 not found at {plot1_source}")
            
            # Process second plot
            plot2_exists = False
            plot2_file = None
            
            if os.path.exists(plot2_source):
                # Try to convert PNG to PDF for better quality
                plot2_pdf_dest = os.path.join(ana_images_dir, "h1o_3x3_summary.pdf")
                if convert_png_to_pdf(plot2_source, plot2_pdf_dest):
                    plot2_file = f"{ana_label}/h1o_3x3_summary.pdf"
                    plot2_exists = True
                    print(f"  Using PDF: {ana_label}/h1o_3x3_summary.pdf")
                else:
                    # Fallback to PNG
                    plot2_png_dest = os.path.join(ana_images_dir, "h1o_3x3_summary.png")
                    shutil.copy2(plot2_source, plot2_png_dest)
                    plot2_file = f"{ana_label}/h1o_3x3_summary.png"
                    plot2_exists = True
                    print(f"  Using PNG: {ana_label}/h1o_3x3_summary.png")
            else:
                print(f"  Warning: Plot 2 not found at {plot2_source}")
            
            # Generate LaTeX for plots
            clean_ana_label = ana_label.replace('-', '_').replace('_', '')
            
            if plot1_exists:
                latex_content += "\\begin{figure}[H]\n"
                latex_content += "    \\centering\n"
                latex_content += f"    \\includegraphics[width=\\textwidth]{{{plot1_file}}}\n"
                latex_content += f"    \\label{{fig:h1o1x3{clean_ana_label}}}\n"
                latex_content += f"    \\caption{{Momentum windows for {safe_ana_label}}}\n"
                latex_content += "\\end{figure}\n\n"
            
            if plot2_exists:
                latex_content += "\\begin{figure}[H]\n"
                latex_content += f"    \\includegraphics[width=\\textwidth]{{{plot2_file}}}\n"
                latex_content += f"    \\label{{fig:h1o3x3{clean_ana_label}}}\n"
                latex_content += f"    \\caption{{Cut parameters for {safe_ana_label}}}\n"
                latex_content += "\\end{figure}\n\n"
        else:
            latex_content += f"\\textbf{{Warning:}} Images directory not found: \\texttt{{{source_images_path.replace('_', '\\_')}}}\n\n"
            print(f"  Warning: Images directory not found: {source_images_path}")
        
        # Start landscape for tables
        latex_content += "\\begin{landscape}\n\n"
        
        # Add cut flow table
        latex_content += "\\subsection{Cut Flow}\n\n"
        cut_flow_path = os.path.join(results_path, "cut_flow.csv")
        latex_content += csv_to_latex_table(
            cut_flow_path,
            caption=f"Cut flow table for {safe_ana_label}",
            label=f"tab:cutflow_{clean_ana_label}",
            output_dir=report_dir,
            use_adjustbox=False
        )
        
        # Add analysis table
        latex_content += "\\subsection{Analysis}\n"
        analysis_path = os.path.join(results_path, "analysis.csv")
        
        if os.path.exists(analysis_path):
            latex_content += csv_to_latex_table(
                analysis_path,
                caption=f"Analysis results for {safe_ana_label}",
                label=f"tab:analysis_{clean_ana_label}",
                output_dir=report_dir,
                use_adjustbox=True
            )
        else:
            latex_content += f"\\textbf{{Warning:}} Analysis results file not found: \\texttt{{{analysis_path.replace('_', '\\_')}}}\n\n"
            print(f"  Warning: Analysis results file not found: {analysis_path}")
        
        # End landscape and add page break
        latex_content += "\\end{landscape}\n"
        latex_content += "\\newpage\n\n"
    
    # Document footer
    latex_content += "\\end{document}\n"
    
    # Write to file
    with open(full_output_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    print(f"LaTeX report generated: {full_output_path}")
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
                print(f"PDF generated successfully: {pdf_path}")
                # Run twice for proper references
                subprocess.run(['pdflatex', latex_filename], 
                             capture_output=True, text=True, cwd=latex_dir)
                print("Second pass completed for proper references")
            else:
                print(f"LaTeX compilation failed:\n{result.stderr}")
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