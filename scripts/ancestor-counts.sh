#!/bin/bash

# Start dataset
DEFNAME=$1 # "nts.mu2e.CosmicCRYSignalAllOnSpillTriggered.MDC2020aw_perfect_v1_3_v06_06_00.root"
OUTPUT_CSV="ancestor-counts-${2}.csv"

# Write CSV header
echo "Dataset,Triggered,Generated,Files,Size" > "$OUTPUT_CSV"

# Get counts for the start dataset
get_counts() {
  local defname="$1"
  local output
  output=$(samDatasetsSummary.sh "$defname")
  local triggered=$(echo "$output" | grep "Triggered:" | awk '{print $2}')
  local generated=$(echo "$output" | grep "Generated:" | awk '{print $2}')
  local files=$(echo "$output" | grep "Files:" | awk '{print $2}')
  local size=$(echo "$output" | grep "Size:" | awk '{print $2}')
  echo "${defname},${triggered},${generated},${files},${size}" >> "$OUTPUT_CSV"
}

# Ancestor loop
CURRENT_DEFNAME="$DEFNAME"
while true; do
  echo "Processing: $CURRENT_DEFNAME"
  get_counts "$CURRENT_DEFNAME"

  FILE=$(samweb list-definition-files "$CURRENT_DEFNAME" | head -n 1)
  [[ -z "$FILE" ]] && break

  ANCESTOR_FILE=$(samweb get-metadata "$FILE" | awk '/Parents:/{found=1} found && /\.(root|art)$/{print $NF; exit}')
  [[ -z "$ANCESTOR_FILE" ]] && break

  ANCESTOR_DEFNAME=$(echo "$ANCESTOR_FILE" | sed 's/\.[0-9_]*\.\([^.]*\)$/.\1/')

  CURRENT_DEFNAME="$ANCESTOR_DEFNAME"
done

echo "Results written to $OUTPUT_CSV"
