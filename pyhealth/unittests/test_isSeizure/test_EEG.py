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


import unittest
import torch
import numpy as np
import os

import torch

# Import component to load and manage the dataset
from pyhealth.datasets import EEG_Dataset, get_dataloader


from pyhealth.models import CNN2D_LSTM_V8

# Import a function that defines how EEG data is processed for seizure detection
from pyhealth.tasks import EEG_isSeizure_fn

class Args:

    """
    This class defines the hyperparameters for the CNN2D_LSTM_V8 model.

    Args:
        num_layers (int): The number of layers in the CNN2D model.
        batch_size (int): The number of samples processed at once.
        dropout (float): The dropout rate for regularization to prevent overfitting.
        num_channel (int): The number of channels in the EEG data.
        output_dim (int): The number of output channels for binary classification (seizure or no seizure).
        enc_model (str): The type of data to process (raw EEG signal).

    """
    def __init__(self):
        self.num_layers = 2     # Number of layers in the CNN2D model
        self.batch_size = 32    # How many samples are processsed at once
        self.dropout = 0.5      # Drop-out rate for regularization to prevent overfitting
        self.num_channel = 1    # Number of channels in the EEG data
        self.output_dim = 2     # Sets number of output channels for binary classification:  seizure or no seizure
        self.enc_model = "raw"  # Sets the type of data to raw.  Note that this code is specifically designed to process raw data

class TestEEGPipeline(unittest.TestCase):

    def setUp(self):
        #  Define paths to a small sample of your TUH EEG data for testing
        #  This method is called before each test method in this class

        self.sample_root_dir = r"C:\Users\bfole\Documents\Computer Science\Python_Projects\PyHealth\pyhealth\unittests\test_isSeizure\Project Data"
        self.output_path = r"C:\Users\bfole\Documents\Computer Science\Python_Projects\PyHealth\pyhealth\unittests\test_isSeizure\test_processed"
        os.makedirs(self.output_path, exist_ok=True)

    def test_dataset_loading(self):

        #  Test EEG_Dataset initialization and data loading
        eeg_dataset = EEG_Dataset(root=self.sample_root_dir, dev=True, refresh_cache=True)
        data = eeg_dataset.process_EEG_data()
        self.assertIsInstance(data, dict)               # Asserts that the data is a dictionary
        self.assertTrue(len(data) > 0)                  # Asserts that some data is loaded
        sample_record = next(iter(data.values()))
        self.assertIn('signal_data', sample_record)     # Asserts that the sample record has 'signal_data' as a key
        self.assertIn('patient_id', sample_record)      # Asserts that the sample record has 'patient_id' as a key

    def test_eeg_is_seizure_fn(self):

        #  Test EEG_isSeizure_fn processing a single record
        eeg_dataset = EEG_Dataset(root=self.sample_root_dir, dev=True, refresh_cache=True)
        data = eeg_dataset.process_EEG_data()
        sample_record = next(iter(data.values()))
        processed_samples = EEG_isSeizure_fn(sample_record)
        self.assertIsInstance(processed_samples, list)      # Asserts that the output is a list
        self.assertTrue(len(processed_samples) > 0)         # Asserts that the list is not empty
        self.assertIn('signal', processed_samples[0])       # Asserts that the first element of the list has
        self.assertIn('label', processed_samples[0])        # signal and label as keys

    def test_model_output_shape(self):
        #  Test CNN2D_LSTM_V8 model output shape

        #  Create an instance of the Args class for passing parameters to the model
        args = Args()

        #  Determine whether a GPU is available or not
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #  Instantiate model
        model = CNN2D_LSTM_V8(args=args, device=device)

        batch_size = 32
        num_channels = 32
        signal_length = 800

        #  Create a dummy input tensor
        #  Set the values for batch_size, num_channels, and signal_length (the shape of the tensor)
        #  Fill the tensor with random numbers
        input_tensor = torch.randn(batch_size, signal_length, num_channels).\
            to(device).float()

        output, _ = model(input_tensor)  # Addressed float error

        #  Assertions about the output shape (adjust these based on your model's expected output)
        self.assertEqual(output.shape, (batch_size, 2))  # Assuming 2 classes (seizure/non-seizure)

    def test_model_training_step(self):
        """Test a single training step with the model"""
        # Setup model
        args = Args()
        device = torch.device("cpu")
        model = CNN2D_LSTM_V8(args=args, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()

        # Create dummy batch with controlled values to ensure loss is significant
        batch_size = 4
        signal_length = 800
        num_channels = 32

        # Create input with correct shape - use ones instead of random to ensure stronger gradients
        input_tensor = torch.ones(batch_size, signal_length, num_channels).float()
        labels = torch.tensor([0, 1, 0, 1])  # Balanced labels

        # Forward pass
        model.train()
        output, _ = model(input_tensor)

        # Print model output and loss before backprop
        print(f"Model output: {output}")

        # Calculate loss
        loss = loss_fn(output, labels)
        print(f"Loss value: {loss.item()}")

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Count parameters with gradients
        params_with_grad = 0
        params_without_grad = 0

        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None and torch.sum(torch.abs(param.grad)) > 0:
                    params_with_grad += 1
                else:
                    params_without_grad += 1
                    print(f"Parameter without gradient: {name}, Shape: {param.shape}")

        print(f"Parameters with gradients: {params_with_grad}")
        print(f"Parameters without gradients: {params_without_grad}")

        # Only test that at least some parameters have gradients
        self.assertTrue(params_with_grad > 0, "No parameters received gradients")

        # Continue with the optimizer step
        optimizer.step()

    def test_model_evaluation(self):
        """Test model evaluation with threshold adjustment"""
        # Setup model
        args = Args()
        device = torch.device("cpu")
        model = CNN2D_LSTM_V8(args=args, device=device)

        # Create dummy batch
        batch_size = 4
        signal_length = 800
        num_channels = 32

        # Create input with correct shape
        input_tensor = torch.randn(batch_size, signal_length, num_channels).float()

        # Set model to evaluation mode
        model.eval()

        # Forward pass
        with torch.no_grad():
            output, _ = model(input_tensor)

            # Get probabilities
            probabilities = torch.softmax(output, dim=1)

            # Test standard argmax prediction
            _, predicted_standard = torch.max(output, 1)
            self.assertEqual(predicted_standard.shape, (batch_size,))

            # Test custom threshold prediction
            threshold = 0.3
            seizure_probs = probabilities[:, 1]
            predicted_custom = (seizure_probs > threshold).int()
            self.assertEqual(predicted_custom.shape, (batch_size,))

if __name__ == '__main__':
    unittest.main()