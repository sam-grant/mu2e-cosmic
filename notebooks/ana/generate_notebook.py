#!/usr/bin/env python3
"""
Script to generate analysis notebooks based on template/template.ipynb
Usage: python generate_notebook.py <ana_label>
"""

import sys
import json
import uuid
from pathlib import Path

from pyutils.pylogger import Logger
logger = Logger(print_prefix="[generate_notebook]", verbosity=2)

def generate_cell_id():
    """Generate a unique cell ID in the format used by Jupyter"""
    return str(uuid.uuid4()).replace("-", "-")

def replace_placeholders(content, ana_label):
    """Replace placeholder strings in the notebook content"""
    replacements = {
        "<ana_label>": ana_label,
        "<n>": ana_label  # Based on the ls command in the template
    }
    
    for placeholder, replacement in replacements.items():
        content = content.replace(placeholder, replacement)
    
    return content

def generate_notebook(ana_label, template_path, output_path=None):
    """
    Generate a new notebook from template with the given ana_label
    
    Args:
        ana_label (str): The analysis label to use
        template_path (str): Path to the template notebook
        output_path (str, optional): Output path. If None, uses ana_label.ipynb
    """
    
    # Extract the prefix before the first underscore
    prefix = ana_label.split('_')[0]
    
    # Set default output path if not provided
    if output_path is None:
        # Create directory path based on prefix
        output_dir = Path(prefix)
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{ana_label}.ipynb"
        logger.log(f"Created/using directory: {output_dir}", "info")
    
    # Read the template notebook
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            notebook = json.load(f)
    except FileNotFoundError:
        logger.log(f"Error: Template file \"{template_path}\" not found.", "error")
        return False
    except json.JSONDecodeError as e:
        logger.log(f"Error: Template file \"{template_path}\" contains invalid JSON: {e}", "error")
        return False
    
    # Process each cell to replace placeholders
    for cell in notebook.get("cells", []):
        # Replace placeholders in markdown cells
        if cell.get("cell_type") == "markdown":
            if "source" in cell:
                if isinstance(cell["source"], list):
                    cell["source"] = [replace_placeholders(line, ana_label) for line in cell["source"]]
                else:
                    cell["source"] = replace_placeholders(cell["source"], ana_label)
        
        # Replace placeholders in code cells
        elif cell.get("cell_type") == "code":
            if "source" in cell:
                if isinstance(cell["source"], list):
                    cell["source"] = [replace_placeholders(line, ana_label) for line in cell["source"]]
                else:
                    cell["source"] = replace_placeholders(cell["source"], ana_label)
        
        # Generate new cell ID to avoid conflicts
        cell["id"] = generate_cell_id()
    
    # Update notebook metadata if needed
    if "metadata" in notebook:
        # You can add any metadata updates here if needed
        pass
    
    # Write the generated notebook
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        logger.log(f"Successfully generated notebook: {output_path}", "success")
        return True
    except Exception as e:
        logger.log(f"Error writing output file: {e}", "error")
        return False

def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) < 2:
        help_message = (
            f"Usage: python generate_notebook.py <ana_label> [template_path] [output_path]\n"
            f"  ana_label: The analysis label to use for replacements\n"
            f"             (prefix before first underscore will be used as directory name)\n"
            f"  template_path: Path to template notebook (default: template/template.ipynb)\n"
            f"  output_path: Output path (default: <prefix>/<ana_label>.ipynb)"
        )
        logger.log(help_message, "info")
        sys.exit(1)
    
    ana_label = sys.argv[1]
    template_path = sys.argv[2] if len(sys.argv) > 2 else "template/template.ipynb"
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Validate ana_label
    if not ana_label or ana_label.strip() == "":
        logger.log("Error: ana_label cannot be empty", "error")
        sys.exit(1)
    
    # Check if template exists
    if not Path(template_path).exists():
        logger.log(f"Error: Template file \"{template_path}\" does not exist", "error")
        sys.exit(1)
    
    # Generate the notebook
    success = generate_notebook(ana_label, template_path, output_path)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()