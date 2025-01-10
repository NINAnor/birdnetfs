#!/bin/bash

# Define the base directory containing the species folders and output directory
BASE_DIR="for_tom/segments"
OUTPUT_DIR="/home/benjamin.cretois/Code/birdnetfs/for_tom/csv"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Header for the CSV file
CSV_HEADER="Filename,Species (BirdNET),BirdNET correct? (Yes/No),If BirdNET incorrect, true species ID or sound source?,How confident are you? (High/Medium/Low),Reason for misidentification? (Related species /Anthropogenic sound/Mimic/No call/Other),Comments"

# Iterate over each species folder in the base directory
for species_folder in "$BASE_DIR"/*; do
    if [ -d "$species_folder" ]; then
        # Get the species name from the folder name
        species_name=$(basename "$species_folder")

        # Define the output CSV file
        csv_file="$OUTPUT_DIR/$species_name.csv"

        # Write the CSV header to the file
        echo "$CSV_HEADER" > "$csv_file"

        # Iterate over the files in the species folder
        for file in "$species_folder"/*; do
            if [ -f "$file" ]; then
                # Get the filename without the path
                filename=$(basename "$file")

                # Add a row to the CSV with the filename and the species name
                echo "$filename,$species_name,,,,,," >> "$csv_file"
            fi
        done
    fi
done

echo "CSV generation complete. Files saved in $OUTPUT_DIR."