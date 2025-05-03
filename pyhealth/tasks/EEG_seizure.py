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

import os
import pickle
import numpy as np


def EEG_isSeizure_fn(record):
    """Processes EEG data for the seizure detection task for a single patient.

    This function takes a dictionary containing EEG data and metadata,
    segments the EEG signal, determines a label for each segment
    (seizure or non-seizure), and saves the processed segments
    to pickle files.

    Args:
        record: A dictionary containing EEG data and metadata for a
            single patient. The dictionary is expected to have the
            following keys:
            - 'patient_id' (str): The unique identifier for the patient.
            - 'edf_file' (str, optional): The name of the original EDF file.
            - 'labels_bi' (list, optional): A list of dictionaries, where
              each dictionary contains information about seizure events
              ('start_time', 'stop_time'). Defaults to an empty list.
            - 'signal_data' (numpy.ndarray): The EEG signal data as a
              2D NumPy array (channels x samples).
            - 'sampling_rate' (int, optional): The sampling rate of the
              EEG data in Hz.
            - 'channel_names' (list, optional): A list of strings
              representing the names of the EEG channels.
            - 'labels_bi': (list, optional): A list of dictionaries, where
              each dictionary contains information about seizure events
              (i.e. 'start_time', 'stop_time', 'label')
            - 'save_path' (str, optional): The directory where processed
              segments should be saved.
            - 'base_name' (str, optional): A base name for the saved
              segment files.

    Returns:
        list: A list of dictionaries, where each dictionary represents
        a processed EEG segment and contains the following keys:
            - 'patient_id' (str): The patient ID.
            - 'record_id' (int): The index of the segment.
            - 'label' (int): The seizure label (1 for seizure, 0 for
              non-seizure).
            - 'epoch_path' (str): The path to the saved pickle file
              containing the segment data.
            - 'signal' (numpy.ndarray): The EEG segment data.

    Example use case:
        This function is called within the `EEG_Dataset.set_task()` method
        to process and prepare EEG data for model training.
    """

    # Check that the record is a dictionary
    if isinstance(record, dict):

    # Extracts the data elements from the record dictionary for a single patient

        # Extract patient ID
        patient_id = record['patient_id']

        # Retrieves the EDF filename
        edf_file = record.get('edf_file')

        # Extracts the raw EEG signal data
        signal_data = record['signal_data']

        # Retrieves the sampling rate (i.e. number of measurements taken per second)
        sampling_rate = record.get('sampling_rate')

        # Retrieves the channel_names
        channel_names = record.get('channel_names')

        # Retrieves the seizure labels
        labels = record.get('labels_bi', [])  # Get labels, default to empty list if not present

        # Gets the path to save processed data
        save_path = record.get('save_path')

        # Gets the base name of the file
        base_name = record.get('base_name')

    else:
        raise TypeError(f"Expected record to be a dictionary, but got a {type(record)} with value: {record}")


    # Creates an empty list to store processed data segments
    samples = []

    # Process and segment the signal data

    # Sets the length of each EEG segment to 800 data points
    segment_length = 800

    # Calculates the number of segments that can be extracted from the signal
    # Assumes that signal_data is a two dimensional array where the second dimension has the length of the EEG recording
    num_segments = signal_data.shape[1] // segment_length

    # Loops through each segment
    for i in range(num_segments):

        # Extracts the i-th EEG segment from signal data
        segment = signal_data[:, i * segment_length: (i + 1) * segment_length].astype(np.float32)

        #  Determine label for this segment

        label = 0  # Default is set to non-seizure

        for event in labels:
            # Get the start and stop times of each event
            start_time = event.get('start_time')
            stop_time = event.get('stop_time')
            event_label = event.get('label')

            # Determine the label for the segment -- 1 for seizure, 0 for non-seizure
            if start_time is not None and stop_time is not None:
                segment_start_time = i * segment_length / sampling_rate
                segment_end_time = (i + 1) * segment_length / sampling_rate

                # Checks whether the current event (seizure) overlaps with the time period of the current EEG segment
                if start_time < segment_end_time and stop_time > segment_start_time:
                    if event_label == "seiz":
                        label = 1  # Seizure detected
                        break
                    elif event_label == "bckg":
                        label = 0

        # Creates the filename for the saved segment, including the base filename and segment index
        epoch_filename = f"{patient_id}_{base_name}_seg_{i}.pkl"

        # Creates the full path to save the file
        epoch_path = os.path.join(save_path, epoch_filename)

        # Saves the segment and its label to a pickle file
        with open(epoch_path, 'wb') as f:
            pickle.dump({'signal': segment, 'label': label}, f)

        # Appends information to 'samples' for a processed segment
        samples.append({
            'patient_id': patient_id,
            'record_id': i,
            'label': label,
            'epoch_path': epoch_path,
            'signal': segment
        })

    return samples



