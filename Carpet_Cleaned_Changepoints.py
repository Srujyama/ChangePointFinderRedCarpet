#!/usr/bin/env python3
# PROGRAM: Carpet Cleaned ChangePoints is a Python3 program that finds different regions in a microbial genome
# Different regions and the specific areas that the change at are provided in a text file
# Region matches are provided in a text file

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

# DATE CREATED: January 4, 2024

# Contrubitors: Srujan S Yamali
# Contrubitors: Arnav Lal
# Contrubitors: Ahmed M Moustafa

# AFFILIATION: Pediatric Infectious Disease Division, Children’s Hospital of Philadelphia,
# Abramson Pediatric Research Center, University of Pennsylvania, Philadelphia,
# Pennsylvania, 19104, USA

# CITATION1: Srujan Yamali, Erin Theiller, Paul Planet, and Ahmed Moustafa
# Redcarpet: A Tool for Rapid Recombination Detection in Staphylococcus aureus and Other Species Amidst Expanding Genomic Databases
# DOI Citation: TBD

# Carpet Cleaned Changepoints: Finds the useful "stains" (squares) in the carpet and compares them to find similar regions
# %%imports
import matplotlib
import time
import matplotlib.pyplot as plt
import pandas as pd
import ruptures as rpt
from multiprocessing import Pool, cpu_count
import numpy as np
import os
from scipy.stats import ttest_ind
from tqdm import tqdm
import argparse
import sklearn
from sklearn.neighbors import KDTree

# %%functions for commandline arguments
# where you add the parameters for your script
"""
num_chunks is just the amount of different chunks you want to split the data into that are then searched for change points.
Once each chunk is processed, the changepoints from all chunks are merged and sorted to provide a global view of the detected changepoints across the dataset.
This sorted list if called all_change_points and is used to visualize the changepoints on the heatmap and line plot.
"""


# python main.py -i /Users/srujanyamali/Downloads/Microbial_Stuff/Microbial_Datasets/GCA_0000270451_ASM2704v1_modified_Redcarpet_report.txt -o /Users/srujanyamali/Downloads/Microbial_Stuff/Microbial_Datasets/


# --file_path: Path to the input file.
# to run the script use the following command (example):
# python main.py -i /path/to/your_dataset.txt -o /path/to/output_directory --num_chunks 1
# make sure you are cd into the directory where the main.py file is located
# the file path should be the path to the dataset you want to analyze
# num_chunks is the number of chunks you want to split the dataset into

def parse_arguments():
    parser = argparse.ArgumentParser(description="Change Point Detection")
    parser.add_argument("-i", "--input_heatmap", type=str, required=True, help="Path to the input heatmap file.")
    parser.add_argument("-o", "--output_directory", type=str, required=True, help="Directory for storing output files.")
    return parser.parse_args()

    parser = argparse.ArgumentParser(description="Change Point Detection")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the input file.")
    parser.add_argument("--num_chunks", type=int, default=10, help="Number of chunks to split the data into.")
    return parser.parse_args()


# function to detect change points in the dataset using the ruptures library
def detect_change_points(data_chunk):
    num_proteins = 1000
        #data_chunk.shape)[0]
    # Scale penalty: base value + a factor of protein count
    penalty = (num_proteins * 1)  # Supposed to be Y = M * X + B but ended up as Y = X (M * X)
    algo = rpt.KernelCPD(kernel="linear").fit(data_chunk)
    return algo.predict(pen=penalty)



# function to calculate pairwise comparisons between all regions in the dataset
def calculate_pairwise_comparisons(data):
    comparisons = {}  # dictionary to store the comparisons
    for i in tqdm(range(data.shape[0]),
                  desc="Calculating Pairwise Comparisons"):  # loop through the data to compare each region with every other region
        for j in range(i + 1, data.shape[0]):  # loop through the data to compare each region with every other region
            t_stat, p_value = ttest_ind(data[i], data[
                j])  # calculate the t-statistic and p-value for the two regions using the ttest_ind function for scipy
            comparisons[(i, j)] = (t_stat, p_value)  # store the t-statistic and p-value in the dictionary
    return comparisons


def pad_to_same_length(arr1,
                       arr2):  # function to pad the arrays to the same length for comparison (if they are not the same length (which they are prob not)
    max_length = max(arr1.shape[1], arr2.shape[1])  # find the max length of the two arrays
    padded_arr1 = np.pad(arr1, ((0, 0), (0, max_length - arr1.shape[1])), mode='constant',
                         constant_values=np.nan)  # pad the first array
    padded_arr2 = np.pad(arr2, ((0, 0), (0, max_length - arr2.shape[1])), mode='constant',
                         constant_values=np.nan)  # pad the second array
    return padded_arr1, padded_arr2


# used lists to store the comparisons and similar regions to avoid the need to store the entire dataset in memory
def find_similar_regions(data, regions, similarity_threshold=0.05, k_neighbors=5):
    """
    Finds similar regions in the dataset using nearest-neighbor search to reduce comparisons.

    Parameters:
    - data (numpy array): The dataset matrix.
    - regions (list of tuples): List of (start, end) indices defining regions.
    - similarity_threshold (float): P-value threshold to determine similarity.
    - k_neighbors (int): Number of closest neighbors to compare (reduces total comparisons).

    Returns:
    - all_region_comparisons (list): All comparisons with p-values.
    - similar_regions (list): Regions that meet similarity criteria.
    """

    # Convert region indices to a KD-Tree for fast nearest-neighbor search
    region_centers = np.array([(r[0] + r[1]) / 2 for r in regions]).reshape(-1, 1)
    tree = KDTree(region_centers, leaf_size=40)

    all_region_comparisons = []
    similar_regions = []
    jaccard_operations = 0

    # Iterate over each region and compare only with nearest neighbors
    for i in tqdm(range(len(regions)), desc="Finding Similar Regions"):
        max_neighbors = min(k_neighbors + 1, len(regions))
        nearest_neighbors = tree.query(region_centers[i].reshape(1, -1), k=max_neighbors, return_distance=False)[0]

        for j in nearest_neighbors:
            if j <= i:  # Avoid duplicate comparisons
                continue

            jaccard_operations += 1
            # Extract region indices
            region1_indices = range(regions[i][0], regions[i][1])
            region2_indices = range(regions[j][0], regions[j][1])

            # Extract data for the two regions
            region1_data = data[:, region1_indices]
            region2_data = data[:, region2_indices]

            # Pad data to same length
            region1_data, region2_data = pad_to_same_length(region1_data, region2_data)

            # Perform t-test
            t_stats, p_values = ttest_ind(region1_data.T, region2_data.T, axis=1, nan_policy='omit')
            avg_p_value = np.nanmean(p_values) if p_values.size > 0 else float('inf')

            # Store all comparisons
            all_region_comparisons.append((regions[i][0], regions[i][1], regions[j][0], regions[j][1], avg_p_value))

            # Store similar regions if they meet the threshold
            if avg_p_value > similarity_threshold:
                similar_regions.append((regions[i][0], regions[i][1], regions[j][0], regions[j][1], avg_p_value))

    return all_region_comparisons, similar_regions, jaccard_operations


def write_change_points_to_file(results, file_path, data_chunk, data_length):
    all_change_points = sorted(
        set().union(*[result[:-1] for result in results]))  # merge and sort the change points from all chunks
    regions = []
    start = 0
    for end in all_change_points:  # split the data into regions based on the change points
        region_data = data_chunk[:, start:end]  # get the data for the region
        regions.append((start, end, region_data))
        start = end + 1
    regions.append((start, data_length, data_chunk[:, start:]))

    with open(file_path, 'w') as file:
        file.write("Detected Regions:\n")
        for i, region in enumerate(regions, start=1):
            file.write(f"Region {i}: {region[0]} to {region[1]}\n")
            file.write("\n")

    return regions

def write_similar_regions_to_file(similar_regions, file_path):
    with open(file_path, 'w') as file:
        file.write("Similar Regions:\n")
        for i, region_pair in enumerate(similar_regions, start=1):
            file.write(
                f"Pair {i}: Region {region_pair[0]}-{region_pair[1]} and Region {region_pair[2]}-{region_pair[3]}, Avg P-value: {region_pair[4]}\n")


def write_all_region_comparisons_to_file(all_region_comparisons, file_path):
    with open(file_path, 'w') as file:
        file.write("All Region Comparisons:\n")
        for i, comparison in enumerate(all_region_comparisons, start=1):
            file.write(
                f"Comparison {i}: Region {comparison[0]}-{comparison[1]} and Region {comparison[2]}-{comparison[3]}, Avg P-value: {comparison[4]}\n")


def merge_adjacent_similar_regions(data, regions, similarity_threshold=0.05):
    merged_regions = []
    i = 0
    while i < len(regions): # Check if there are more regions to process
        start1, end1, data1 = regions[i] # Get the current region

        while i + 1 < len(regions): # Check if there is a next region
            start2, end2, data2 = regions[i + 1] # Get the next region

            if end1 + 1 == start2: # Check if regions are adjacent
                region1_data, region2_data = pad_to_same_length(data1, data2) # Pad the data to the same length
                t_stats, p_values = ttest_ind(region1_data.T, region2_data.T, axis=1, nan_policy='omit')
                avg_p_value = np.nanmean(p_values) if p_values.size > 0 else float('inf')

                if avg_p_value > similarity_threshold:
                    print(f"Merging regions ({start1}, {end1}) and ({start2}, {end2}), p-value: {avg_p_value:.4f}")
                    end1 = end2
                    i += 1
                    continue
            break

        merged_regions.append((start1, end1))
        i += 1

    return merged_regions


def save_merged_regions_to_folder(merged_regions, output_directory):
    merged_folder = os.path.join(output_directory, "merged_regions")
    os.makedirs(merged_folder, exist_ok=True)  # Create folder if not exists

    merged_file_path = os.path.join(merged_folder, "merged_change_points.txt")
    with open(merged_file_path, 'w') as f:
        f.write("Merged Regions:\n")
        for idx, (start, end) in enumerate(merged_regions, 1):
            f.write(f"Region {idx}: {start} to {end}\n")

    print(f"Merged regions saved to {merged_file_path}")


def export_merged_heatmap(data, merged_regions, output_directory):
    merged_folder = os.path.join(output_directory, "merged_regions")
    os.makedirs(merged_folder, exist_ok=True)

    plt.figure(figsize=(20, 20))
    plt.imshow(data, cmap='hot', interpolation='nearest')

    for start, end in merged_regions:
        plt.axvline(x=end, color='cyan', linestyle='-')
        plt.axhline(y=end, color='cyan', linestyle='-')

    plt.title('Heatmap with Merged Change Points')
    plt.ylabel('Protein [ordered]')
    plt.xlabel('Protein [ordered]')

    heatmap_file_path = os.path.join(merged_folder, "merged_heatmap.png")
    plt.savefig(heatmap_file_path, format='png')
    plt.close()

    print(f"Merged heatmap saved to {heatmap_file_path}")


def process_chunk_parallel(data_chunk):
    return detect_change_points(data_chunk)


def main():
    start_time = time.time()  # Track total execution time

    args = parse_arguments()

    try:
        input_heatmap = args.input_heatmap
        output_directory = args.output_directory

        redcarpet = pd.read_csv(input_heatmap, sep='\t')  # load the dataset
        redcarpet_npy = redcarpet.to_numpy().astype('float')  # convert the dataset to a numpy array
        num_proteins = redcarpet_npy.shape[0]  # Number of proteins

        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    data_length = redcarpet_npy.shape[1]

    data_chunks = [redcarpet_npy]  # split the data into chunks

    try:
        print("Processing the dataset as a single chunk in parallel...")

        with Pool(cpu_count()) as pool:
            results = list(pool.map(process_chunk_parallel, data_chunks))  # process the data chunks in parallel

        print("Dataset processed.")
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return

    plt.figure(figsize=(20, 20))  # plot the heatmap of the dataset with the detected change points

    plt.imshow(redcarpet_npy, cmap='hot', interpolation='nearest')
    all_change_points = sorted(set().union(*[result[:-1] for result in results]))  # merge and sort change points
    for bkpt in all_change_points:
        plt.axvline(x=bkpt, color='cyan', linestyle='-')  # add vertical lines for change points
        plt.axhline(y=bkpt, color='cyan', linestyle='-')  # add horizontal lines for change points
    plt.title('Visualization of Protein Changepoints with Detected Change Points')
    plt.ylabel('Protein [ordered]')
    plt.xlabel('Protein [ordered]')
    heatmap_output_file = os.path.join(output_directory, "heatmap_visualization_with_lines.png")
    plt.savefig(heatmap_output_file, format='png')
    print(f"Heatmap with lines saved as: {heatmap_output_file}")
    plt.clf()  # clear the figure after saving
    # plot the heatmap of the dataset

    all_change_points = sorted(
        set().union(*[result[:-1] for result in results]))  # merge and sort the change points from all chunks
    for bkpt in all_change_points:
        plt.axvline(x=bkpt, color='cyan', linestyle='-')
        plt.axhline(y=bkpt, color='cyan', linestyle='-')

    plt.title('Visualization of Protein Changepoints with Detected Change Points')
    plt.ylabel('Protein [ordered]')
    plt.xlabel('Protein [ordered]')

    plot_data = redcarpet_npy[0, :]  # plot the line plot of the dataset with the detected change points

    plt.figure(figsize=(40, 15))

    plt.plot(plot_data, lw=1)
    for bkpt in all_change_points:
        if (bkpt < len(plot_data)):
            plt.axvline(x=bkpt, color='orange', linestyle='-', linewidth=2)
    plt.title('Visualization of Change Points')
    plt.ylabel('Value')
    plt.xlabel('Index')
    lineplot_output_file = os.path.join(output_directory, "line_plot_visualization_with_lines.png")
    plt.savefig(lineplot_output_file, format='png')
    print(f"Line plot with lines saved as: {lineplot_output_file}")
    plt.clf()  # clear the figure after saving

    for bkpt in all_change_points:
        if (bkpt < len(plot_data)):
            plt.axvline(x=bkpt, color='orange', linestyle='-', linewidth=2)

    plt.title('Visualization of Change Points')
    plt.ylabel('Value')
    plt.xlabel('Index')

    print("Change points visualization completed.")

    # The following three lines write the change points and regions to a file in the same directory as the dataset
    directory = output_directory
    change_points_file_path = os.path.join(directory, 'change_points.txt')
    regions = write_change_points_to_file(results, change_points_file_path, redcarpet_npy, data_length)
    print("\nChange points and regions written to file.")

    print("Finding similar regions using comparisons...")
    # The following 5 lines calculate the pairwise comparisons between all regions in the dataset and write the similar regions and all region comparisons to a file in the same directory as the dataset
    all_region_comparisons, similar_regions, jaccard_operations = find_similar_regions(redcarpet_npy, regions)
    similar_regions_file_path = os.path.join(directory, 'similar_regions.txt')
    write_similar_regions_to_file(similar_regions, similar_regions_file_path)
    all_region_comparisons_file_path = os.path.join(directory, 'all_region_comparisons.txt')
    write_all_region_comparisons_to_file(all_region_comparisons, all_region_comparisons_file_path)
    print("Similar regions and all region comparisons written to file.")


    total_time = time.time() - start_time

    # Write execution time and dataset info to Information.txt
    info_file_path = os.path.join(directory, "Information.txt")
    with open(info_file_path, 'w') as info_file:
        info_file.write(f"Total Execution Time: {total_time:.2f} seconds\n")
        info_file.write(f"Number of Proteins: {num_proteins}\n")
        info_file.write(f"Jaccard Similarity Operations: {jaccard_operations}\n")

    print(f"\nTotal Execution Time: {total_time:.2f} seconds")
    print(f"Information written to {info_file_path}")

    print("\nChecking adjacent regions for similarity and merging...")

    print("\nMerging adjacent similar regions...")
    merged_regions = merge_adjacent_similar_regions(redcarpet_npy, regions)

    save_merged_regions_to_folder(merged_regions, output_directory)
    export_merged_heatmap(redcarpet_npy, merged_regions, output_directory)

    merged_regions_file_path = os.path.join(directory, 'merged_change_points.txt')
    with open(merged_regions_file_path, 'w') as f:
        f.write("Merged Regions:\n")
        for idx, (start, end) in enumerate(merged_regions, 1):
            f.write(f"Region {idx}: {start} to {end}\n")

    print(f"Merged regions written to {merged_regions_file_path}")



if __name__ == "__main__":
    main()
