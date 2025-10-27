#!/bin/bash
set -e

clear

# List of all LaTeX files
FILES=("syntactic-categories" "example-program" "judgments-and-rules" "context-validity" "range-analysis" "backprop" "backprop-range" "substructural-rules")

# Function to convert a single file
convert_file() {
    local file=$1
    echo "Processing $file.tex..."
    pdflatex -halt-on-error "$file.tex"
    magick -density 300 "$file.pdf" -background white -alpha remove -quality 90 -trim -bordercolor white -border 20 -strip "$file.png"
    echo "Created $file.png"
}

# If argument provided, convert only that file
if [ $# -eq 1 ]; then
    if [ -f "$1.tex" ]; then
        convert_file "$1"
    else
        echo "Error: $1.tex not found"
        exit 1
    fi
else
    # Convert all files
    for file in "${FILES[@]}"; do
        convert_file "$file"
    done
fi

echo "success."
