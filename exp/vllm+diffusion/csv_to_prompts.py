#!/usr/bin/env python3
"""
Script to convert VidProM CSV file to a text file with one prompt per line.
"""

import csv
import argparse
from pathlib import Path


def csv_to_prompts(csv_file: str, output_file: str, max_prompts: int = None):
    """
    Extract prompts from CSV file and save to text file.

    Args:
        csv_file: Path to input CSV file
        output_file: Path to output text file
        max_prompts: Maximum number of prompts to extract (None for all)
    """
    input_path = Path(csv_file)
    output_path = Path(output_file)

    if not input_path.exists():
        print(f"Error: Input file '{csv_file}' not found.")
        return

    prompts_written = 0

    try:
        with (
            open(input_path, "r", encoding="utf-8") as infile,
            open(output_path, "w", encoding="utf-8") as outfile,
        ):
            reader = csv.DictReader(infile)

            for row in reader:
                if max_prompts and prompts_written >= max_prompts:
                    break

                prompt = row["prompt"].strip()
                if prompt:  # Skip empty prompts
                    # Replace various line terminators with \n escape sequences
                    # Handle: newline, carriage return, Line Separator, Paragraph Separator
                    clean_prompt = (
                        prompt.replace("\n", "\\n")  # Regular newline
                        .replace("\r", "")  # Carriage return
                        .replace("\u2028", "\\n")  # Line Separator (LS)
                        .replace("\u2029", "\\n")  # Paragraph Separator (PS)
                        .replace("\x0b", "\\n")  # Vertical tab
                        .replace("\x0c", "\\n")
                    )  # Form feed
                    outfile.write(clean_prompt + "\n")
                    prompts_written += 1

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return

    print(f"Successfully extracted {prompts_written} prompts from '{csv_file}'")
    print(f"Output saved to '{output_file}'")


def main():
    parser = argparse.ArgumentParser(
        description="Convert VidProM CSV to text file with one prompt per line"
    )

    parser.add_argument("csv_file", help="Input CSV file path")
    parser.add_argument(
        "-o",
        "--output",
        default="prompts.txt",
        help="Output text file path (default: prompts.txt)",
    )
    parser.add_argument(
        "-n", "--max-prompts", type=int, help="Maximum number of prompts to extract"
    )

    args = parser.parse_args()

    csv_to_prompts(args.csv_file, args.output, args.max_prompts)


if __name__ == "__main__":
    main()
