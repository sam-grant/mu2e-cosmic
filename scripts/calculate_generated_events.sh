#!/bin/bash
#
# Simple version is:
#    mu2einit
#    muse setup SimJob
#    samDatasetsSummary.sh defname
#    
# Calculate number of generated events in dataset
# IMPORTANT: Run setup first: source setup_mu2e_env.sh && muse setup SimJob

set -e  # Exit on error

# Check if samweb is available
check_environment() {
    if ! command -v samweb &> /dev/null; then
        echo "Error: Please run setup first."
        echo "source setup_mu2e_env.sh && muse setup SimJob"
        exit 1
    fi
}

# Function to get first file from dataset
get_first_file() {
    local def="$1"
    local file=$(samweb list-definition-files "$def" 2>/dev/null | head -n 1)
    if [[ -z "$file" ]]; then
        echo "Error: No files found in dataset '$def'" >&2
        exit 1
    fi
    echo "$file"
}

# Find DTS ancestor file
find_dts_ancestor() {
    local first_file="$1"
    local ancestors=$(samweb list-files "isancestorof: (file_name \"$first_file\")")
    local dts_file=$(echo "$ancestors" | grep "^dts\." | grep "MDC2020ar")
    
    if [[ -z "$dts_file" ]]; then
        local dts_file=$(echo "$ancestors" | grep "^dts\." | head -n 1) # first one
    fi

    if [[ -z "$dts_file" ]]; then
        echo "Error: No dts ancestor found"
        exit 1
    fi

    echo "$dts_file"
}

# Extract dataset name (remove run/subrun numbers)
extract_dataset_name() {
    local filename="$1"
    echo "$filename" | sed 's/\.[0-9]\{6\}_[0-9]\{8\}\.art$/.art/'
}

# Parse samDatasetsSummary output and format as CSV
parse_summary_output() {
    local defname="$1"
    local dts_dataset="$2"
    local output=$(samDatasetsSummary.sh "$dts_dataset" 2>/dev/null)
    
    local files=$(echo "$output" | grep "^Files:" | awk '{print $2}')
    local triggered=$(echo "$output" | grep "^Triggered:" | awk '{print $2}')
    local generated=$(echo "$output" | grep "^Generated:" | awk '{print $2}')
    
    echo "$defname,$dts_dataset,$files,$triggered,$generated"
}

# Process a single dataset
process_dataset() {
    local defname="$1"
    
    local first_file=$(get_first_file "$defname")
    local dts_file=$(find_dts_ancestor "$first_file")
    local dts_dataset=$(extract_dataset_name "$dts_file")
    
    # Output CSV row
    parse_summary_output "$defname" "$dts_dataset"
}

# Main function
main() {
    local input_file="${1:-defnames.txt}"
    
    if [[ ! -f "$input_file" ]]; then
        echo "Error: Input file '$input_file' not found" >&2
        echo "Usage: $0 [input_file]" >&2
        echo "Default input file: defnames.txt" >&2
        exit 1
    fi
    
    check_environment
    
    # Print CSV header
    echo "defname,ancestor,files,triggered,generated"
    
    # Process each dataset
    while IFS= read -r defname || [[ -n "$defname" ]]; do
        # Skip empty lines and comments
        [[ -z "$defname" || "$defname" =~ ^[[:space:]]*# ]] && continue
        
        # Strip leading/trailing whitespace
        defname=$(echo "$defname" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        
        if [[ -n "$defname" ]]; then
            process_dataset "$defname"
        fi
    done < "$input_file"
}

# Run main function
main "$@"
