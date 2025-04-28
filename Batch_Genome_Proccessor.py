#!/usr/bin/env python3
# PROGRAM: Cluster Carpet Cleaner is a program that allows Carpet Cleaned ChangePoints to be used for batch processing

# Copyright (C) 2025 Srujan S. Yamali

#########################################################################################
# LICENSE
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#########################################################################################

# DATE CREATED: April 8, 2025

# %%imports
import os
import argparse
import subprocess

# example use case: python3 script.py /path/to/input /path/to/output /path/to/your_script.py

def run_script_on_file(input_file, output_directory, script_path):
    try:
        file_name = os.path.splitext(os.path.basename(input_file))[0]
        genome_output_dir = os.path.join(output_directory, file_name)

        if not os.path.exists(genome_output_dir):
            os.makedirs(genome_output_dir)

        command = ["python3", script_path, "-i", input_file, "-o", genome_output_dir]
        subprocess.run(command, check=True)
        print(f"Successfully processed: {input_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {input_file}: {e}")

def process_folder(input_folder, output_directory, script_path):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    input_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    for file in input_files:
        run_script_on_file(file, output_directory, script_path)

def main():
    parser = argparse.ArgumentParser(description="Process files or folders with a specified script.")
    parser.add_argument("input_path", type=str, help="Path to the input file or folder")
    parser.add_argument("output_directory", type=str, help="Path to the output directory")
    parser.add_argument("script_path", type=str, help="Path to the script to be executed")

    args = parser.parse_args()

    if os.path.isdir(args.input_path):
        print(f"Processing folder: {args.input_path}")
        process_folder(args.input_path, args.output_directory, args.script_path)
    elif os.path.isfile(args.input_path):
        print(f"Processing single file: {args.input_path}")
        run_script_on_file(args.input_path, args.output_directory, args.script_path)
    else:
        print("Invalid input path. Please provide a valid file or folder.")

if __name__ == "__main__":
    main()
