#!/bin/bash

# Script to add a line to the process block in YAML files

# Usage:
# /path/to/script/modify_yaml_files.sh "generated_events: 41100000000" /path/to/yaml/files/*.yaml

set -e

# Function to add line to process block
add_line_to_process() {
    local yaml_file="$1"
    local new_line="$2"
    
    echo "Processing: $yaml_file"
    
    # Create backup
    cp "$yaml_file" "$yaml_file.bak"
    
    # Use awk to add the line at the end of the process block
    awk -v new_line="$new_line" '
    /^process:/ { in_process = 1 }
    /^[a-zA-Z]/ && !/^process:/ && in_process { 
        # We hit a new top-level block, so add our line before it
        print "    " new_line
        in_process = 0 
    }
    { print }
    END { 
        # If we were still in process block at end of file
        if (in_process) {
            print "    " new_line
        }
    }
    ' "$yaml_file" > "$yaml_file.tmp"
    
    mv "$yaml_file.tmp" "$yaml_file"
    echo "  Added: $new_line"
}

# Main function
main() {
    local line_to_add="$1"
    
    if [[ $# -eq 0 ]]; then
        echo "Usage: $0 'line_to_add' [directory_or_files...]"
        echo "Example: $0 'verbosity: 3' /path/to/yaml/files"
        echo "Example: $0 'verbosity: 3' *.yaml"
        echo "Example: $0 'max_files: 1000' /home/user/configs/"
        echo ""
        echo "If no directory/files specified, will process all *.yaml files in current directory"
        exit 1
    fi
    
    # If no files/directory specified after the line, use current directory
    if [[ $# -eq 1 ]]; then
        set -- "$1" .
    fi
    
    shift  # Remove the line_to_add from arguments
    
    # Build file list
    local files=()
    for arg in "$@"; do
        if [[ -d "$arg" ]]; then
            # It's a directory, add all yaml files from it
            while IFS= read -r -d '' file; do
                files+=("$file")
            done < <(find "$arg" -maxdepth 1 -name "*.yaml" -print0)
        elif [[ -f "$arg" ]]; then
            # It's a file
            files+=("$arg")
        else
            echo "Warning: Not found: $arg"
        fi
    done
    
    if [[ ${#files[@]} -eq 0 ]]; then
        echo "No YAML files found!"
        exit 1
    fi
    
    
    echo "Adding line: '$line_to_add'"
    echo "Processing ${#files[@]} files..."
    echo ""
    
    for yaml_file in "${files[@]}"; do
        if [[ -f "$yaml_file" ]]; then
            add_line_to_process "$yaml_file" "$line_to_add"
        else
            echo "Warning: File not found: $yaml_file"
        fi
    done
    
    echo ""
    echo "Done! Backups created with .bak extension"
}

main "$@"