"""
Names:          Brendan Foley and Brian Gabriel
NetIDs:         bfole2, brian11
Paper Title:    Real-Time Seizure Detection using EEG A Comprehensive Comparison of Recent Approaches under a Realistic Setting
Paper Link:     https://arxiv.org/abs/2201.08780
Description:    We are using PyHealth to enable the use of the CNN2D_LSTM model against raw EEG signal data.  This model should
                be applied against the v.2.0.3 dataset (the paper developed their models and trained against the v.1.5.2 dataset
                which is no longer available).

                The dataset is available here: https://isip.piconepress.com/projects/nedc/html/tuh_eeg/.  You will need to request
                access to the dataset by completing the form as described on the page.

                The reason we used the CNN2D_LSTM model is that it performed the best overall of the models in the paper.
                The CNN2D + LSTM model performed comparably with ResNet18 models but was faster and met the real-time constraint
                (i.e., process each window within its shift length).

"""


# Provides functions for interacting with the OS, such as creating directories,
# listing files, and joining paths

from scipy.signal import resample

import os

# Imports the mne library designed for processing EEG data in EDF format
import mne

# For working with DataFrames
import pandas as pd

# Used for finding files based on patterns, e.g. finding files such as .edf
import glob

import numpy as np

from pyhealth.datasets import BaseSignalDataset


class EEG_Dataset(BaseSignalDataset):
    """
    This class is a subclass of BaseSignalDataset designed for loading and
    processing EEG data from the TUH EEG Corpus.  It handles the directory
    structure, reads EDF files, and associates them with corresponding
    metadata from CSV files.

    Args:
        root (str): The root directory of the TUH EEG Corpus.
        dev (bool, optional): If True, loads a smaller subset of the data
            for development purposes. Defaults to False.
        refresh_cache (bool, optional): If True, reprocesses the dataset
            from scratch, updating the cached version. Defaults to False.
    """
    def __init__(self, root, dev=False, refresh_cache=False):

        # Calls the constructor of the BaseSignalDataset
        super().__init__(dataset_name="EEG_Dataset", root=root,
                         dev=dev, refresh_cache=refresh_cache)

        # Creates a filepath for saving the processed data
        self.save_path = os.path.join(self.root, "processed_data")

        # Creates the directory if it doesn't exist for saving the above file
        os.makedirs(self.save_path, exist_ok=True)

    def process_EEG_data(self):
        """
        Processes EEG data from the TUH EEG Corpus, loading EDF files and
        associating them with metadata from CSV files.  It organizes the
        data into a dictionary where keys are record identifiers and
        values are dictionaries containing the EEG signal, sampling rate,
        channel names, and seizure labels.

        Args:
            None

        Returns:
            dict: A dictionary where keys are record identifiers
                  (e.g., "patient_id_basename_index") and values are
                  dictionaries with the following keys:
                - "patient_id" (str): The patient ID.
                - "edf_file" (str): Path to the EDF file.
                - "csv_bi_file" (str): Path to the CSV_BI file.
                - "signal_data" (numpy.ndarray): The EEG signal data
                  (channels x samples).
                - "sampling_rate" (float): The sampling rate of the EEG data.
                - "channel_names" (list): A list of channel names.
                - "labels_bi" (list): A list of dictionaries, where each
                  dictionary contains information about seizure events
                  ('start_time', 'stop_time', 'label').
                - "save_path" (str): The path to save processed data.
                - "base_name" (str): The base name of the file.

        """

        # Creates an empty dictionary for loading and processing the EEG data
        patients = {}

        # Iterates through the directories under the root directory (i.e. train, eval, dev)
        # dataset_split is the current directory for review
        for dataset_split in os.listdir(self.root):

            # Skip the processed_data directory which will be created for saving the file
            if dataset_split == "processed_data":
                continue

            # Construct the full path to the current dataset split directory
            dataset_split_path = os.path.join(self.root, dataset_split)

            # Check if the above path is a directory. If it isn't skip
            if not os.path.isdir(dataset_split_path):
                continue

            # Iterate through the nested directory structure, which is organized as
            # root/dataset_split/patient_dir/session_dir/montage_dir/
            for patient_dir in os.listdir(dataset_split_path):

                # Construct the full path to a patient's directory
                patient_path = os.path.join(dataset_split_path, patient_dir)
                if not os.path.isdir(patient_path):
                    continue

                # The patient_id is the name of the patient directory
                patient_id = patient_dir

                # Creating a record in the patients dictionary for patient_id
                patients[patient_id] = []

                # Going deeper into the directory structure to process data at montage level
                for session_dir in os.listdir(patient_path):
                    session_path = os.path.join(patient_path, session_dir)
                    if not os.path.isdir(session_path):
                        continue
                    for montage_dir in os.listdir(session_path):
                        montage_path = os.path.join(session_path, montage_dir)
                        if not os.path.isdir(montage_path):
                            continue

                        # Find all of the files with the .edf extension in the current
                        # montage directory and sorts them
                        edf_files = sorted(glob.glob(os.path.join(montage_path, '*.edf')))

                        # Find all files with the .csv_bi extension
                        csv_bi_files = sorted(glob.glob(os.path.join(montage_path, '*.csv_bi')))  # Load binary labels

                        # Iterate through each .edf

                        for i, edf_file in enumerate(edf_files):

                            # Extract the file name without the extension, splitting the filename and the extension,
                            # taking the name part
                            base_name = os.path.splitext(os.path.basename(edf_file))[0]

                            # Find the .csv_bi file that has the same base name as the .edf file
                            # Iterate through csv_bi_files to find the .csv_bi file with the same base name
                            csv_bi_file = next((f for f in csv_bi_files if base_name in f), None)

                            try:
                                # Reads the .edf file.  Preload:  Loads all data into memory
                                raw = mne.io.read_raw_edf(edf_file, preload=True, verbose='ERROR')

                                # Extracts the raw EEG signal data as a NumPy array.
                                signal_data = raw.get_data()

                                # Check the number of channels
                                num_channels = signal_data.shape[0]

                                if num_channels > 32:
                                    # Select the first 32 channels.  The issue with the data is that the number of channels varied:  31, 32, and 40+.
                                    # We needed to ensure a consistent number of channels
                                    signal_data = signal_data[:32, :]

                                elif num_channels < 32:
                                    # Pad with zeros to make 32 channels
                                    padding_needed = 32 - num_channels
                                    signal_data = np.pad(signal_data, ((0, padding_needed), (0, 0)), 'constant')

                                # Get the sampling frequency
                                sampling_rate = raw.info['sfreq']

                                # Set sampling rate to 200 Hz, the same as in the academic paper

                                # --- Resampling Code ---
                                common_sampling_rate = 200  # Hz
                                if sampling_rate != common_sampling_rate:
                                    num_samples = int(signal_data.shape[1] * common_sampling_rate / sampling_rate)
                                    signal_data = resample(signal_data, num=num_samples, axis=1)
                                    sampling_rate = common_sampling_rate

                                # Get the names of the EEG channels
                                channel_names = raw.ch_names

                                # Initialize labels_bi for storing the binary labels
                                labels_bi = None

                                # If there's a csv_bi_file
                                if csv_bi_file:
                                    try:
                                        # Read the file into a pandas DataFrame.
                                        # Skips the first five rows which are commentary
                                        labels_bi_df = pd.read_csv(csv_bi_file, skiprows=5)

                                        # Convert the DataFrame to a list of dictionaries
                                        labels_bi = labels_bi_df[['start_time', 'stop_time', 'label']].to_dict(
                                            orient='records')

                                    except Exception as e:
                                        print(f"Error reading CSV_BI file {csv_bi_file}: {e}")

                                # Create the path where .pkl files will be stored
                                save_path = os.path.join(self.root, "processed_data")

                                # Create a unique key for each record
                                record_key = f"{patient_id}_{base_name}_{i}"  # i is the index from enumerate

                                # For debugging
                                # print(f"Record Key: {record_key}")

                                patients[record_key] = {  # Use the unique key
                                    "patient_id": patient_id,
                                    "edf_file": edf_file,
                                    "csv_bi_file": csv_bi_file,
                                    "signal_data": signal_data,
                                    "sampling_rate": sampling_rate,
                                    "channel_names": channel_names,
                                    "labels_bi": labels_bi,
                                    "save_path": save_path,
                                    "base_name": base_name
                                }


                                # Ensure that we remove any potential list
                                if patient_id in patients:
                                    del patients[patient_id]

                            except Exception as e:
                                print(f"Error processing EDF file {edf_file}: {e}")

        return patients




