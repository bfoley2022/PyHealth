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

import torch

# Import component to load and manage the dataset
from pyhealth.datasets import EEG_Dataset

# Import components to split the dataset for training and validation, and to feed data to the model in batches
from pyhealth.datasets import split_by_visit, get_dataloader

from pyhealth.models import CNN2D_LSTM_V8

# Import a function that defines how EEG data is processed for seizure detection
from pyhealth.tasks import EEG_isSeizure_fn

import numpy as np

# Import function that can be used to pad sequences (i.e. EEG signals) so they have the same length,
# needed for batch processing
from torch.nn.utils.rnn import pad_sequence

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix


# Defining values for args and device in order for the CNN2D_LSTM_V8 to work
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


#  Function that pads a batch of EEG signals, because EEG recordings can have different lengths
def pad_batch(batch_signals, max_len, batch_size, device):
    """Pads a batch of EEG signals to the maximum length in the batch.

    This function takes a list of EEG signals, determines the maximum length
    among them, and pads each signal with zeros to match this maximum length.
    This is necessary for efficient batch processing of signals with varying lengths.

    Args:
        batch_signals (list): A list of EEG signals, where each signal is a NumPy array.
                              Each signal has dimensions (num_channels, signal_length).
        max_len (int): The maximum length to which the signals should be padded.
        batch_size (int): The desired batch size.  Padding will occur to make sure the output tensor's
                        first dimension matches the batch_size.
        device (torch.device): The PyTorch device (CPU or GPU) where the padded tensor
                               will be stored.

    Returns:
        torch.Tensor: A PyTorch tensor containing the padded EEG signals.
                      The tensor has dimensions (batch_size, num_channels, max_len).

    Example use case:
        This function is used within the training and evaluation loops to preprocess
        batches of EEG signals before they are fed into the CNN2D_LSTM_V8 model.
    """

    # Convert each signal in the batch to a PyTorch tensor and transpose it
    signals_for_padding = [torch.tensor(signal).transpose(0, 1) for signal in batch_signals]
    padded_signals_list = []

    for signal_tensor in signals_for_padding:
        # Calculate how much padding is needed to make the current signal have max length
        # (i.e. where max_length is the max length of a signal in a batch)
        # Add tensors to padded_signals_list

        padding_needed = max_len - signal_tensor.shape[1]
        if padding_needed > 0:
            padded_signal = torch.nn.functional.pad(signal_tensor, (0, padding_needed), "constant", 0)

            padded_signals_list.append(padded_signal)
        else:
            padded_signals_list.append(signal_tensor)


    # Pads the list of tensors to the longest sequence length in the batch
    # Move the tensor to the specified device
    padded_signals = pad_sequence(padded_signals_list, batch_first=True).transpose(1, 2).float().to(device)

    # Handle batches smaller than desired batch_size
    if padded_signals.shape[0] < batch_size:
        padding_needed = batch_size - padded_signals.shape[0]
        padded_signals = torch.nn.functional.pad(padded_signals, (0, 0, 0, 0, 0, padding_needed), 'constant', 0)

    # Returns the list of tensors with padding
    return padded_signals

#  This block of code runs when this Python code is executed
if __name__ == '__main__':

    #  **************************************
    #  ****  STEP 1:  Load Signal Data  *****
    #  **************************************

    #  Create an instance of EEG_Dataset to load the EEG data from the specified root directory
    dataset = EEG_Dataset(root=r"C:\Users\bfole\Documents\Computer Science\University of Illinois Urbana-Champaign\Deep Learning for Healthcare\Project\Project Data",
                            dev=True,
                            refresh_cache=True)

    #  ******************************
    #  ****  STEP 2:  Set Task  *****
    #  ******************************

    #  Apply function to prepare the data for the seizure detection task
    EEG_Dataset_ds = dataset.set_task(EEG_isSeizure_fn)

    EEG_Dataset_ds.stat()

    # Split the data into training, validation, and test data sets based on specified rations
    train_dataset, val_dataset, test_dataset = split_by_visit(EEG_Dataset_ds, [0.6, 0.2, 0.2])


    train_dataloader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

    #  Create an instance of the Args class for passing parameters to the model
    args = Args()

    #  Determine whether a GPU is available or not
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #  Instantiate model
    model = CNN2D_LSTM_V8(args=args, device=device)

    #  *********************************
    #  ****  STEP 3:  Train Model  *****
    #  *********************************

    #  Used to change the weights and learning rate in order to reduce losses
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_labels = []
    for batch in train_dataloader:
        train_labels.extend(batch['label'])

    #  Define loss function - quantifies the difference between actual class label and predicted probabilities in the model
    loss_fn = torch.nn.CrossEntropyLoss()

    #  Sets the number of epochs, or complete passes through training data
    num_epochs = 5

    for epoch in range(num_epochs):

        #  Set model to training mode
        model.train()

        #  Initialize hidden state at the start of training
        #  Start each training epoch with a fresh hidden state
        hidden = None

        #  Loop through each batch of data provided by train_dataloader
        for i, batch in enumerate(train_dataloader):
            print(f"\n--- Batch {i}: Training ---\n")

            # 1. --- Pad the batch ---
            # Calculate the maximum sequence length in the current batch
            max_len = max(signal.shape[1] for signal in batch['signal'])

            # Pads the EEG signals in the batch to the max length
            padded_signals = pad_batch(batch['signal'], max_len, args.batch_size, device)

            # Converts labels to a PyTorch tensor
            labels = torch.tensor(batch['label']).to(device)

            # --- Forward and backward pass ---

            # Send padded signals to the model to get output predictions
            # Pass the hidden state between batches during training
            output, hidden = model(padded_signals)

            # Detach hidden state to prevent back propagation through the entire sequence history
            if hidden is not None:
                hidden = (hidden[0].detach(), hidden[1].detach())

            # Perform backwards propagation and optimization

            # Calculate the loss -- how poorly the model's predictions match the actual correct labels
            loss = loss_fn(output, labels)

            # Prevent gradients from accumulating between training iterations
            optimizer.zero_grad()

            # Calculate the gradient of loss with respect to all of the model's parameters that contributed to output
            # i.e. determine how much each parameter needs to change to reduce loss
            loss.backward()

            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item()

            # Update model parameters based on the gradients calculated by loss.backwards()
            optimizer.step()

            print(f"--- Completed for Batch {i} ---")

            print(f"Pred values: {torch.argmax(output, dim=1)}")
            print(f"True labels: {labels}\n")

            if i == 14:
                break

        print(f'Epoch {epoch + 1}/{num_epochs}')

    #  ************************************
    #  ****  STEP 4:  Evaluate Model  *****
    #  ************************************

    # Set model to evaluation mode
    model.eval()

    # Start the evaluation phase with a completely fresh hidden state
    hidden = None

    # Initialize variables
    total_correct = 0
    total_samples = 0
    all_predicted = []
    all_labels = []
    all_probabilities = []

    # Disable gradient calculations (helps save memory and speed up evaluation since we don't need to update the model
    with torch.no_grad():

        for i, batch in enumerate(test_dataloader):
            print(f"\n--- Batch {i}: Evaluation ---")

            # --- Pad the batch (using the same function as above)---
            max_len = max(signal.shape[1] for signal in batch['signal'])
            padded_signals_test = pad_batch(batch['signal'], max_len, args.batch_size, device)

            # Get the test labels
            labels_eval = torch.tensor(batch['label']).to(device)

            # Get the number of samples in the batch
            num_samples_in_batch = padded_signals_test.shape[0]
            # print(f"  Batch {i}: num_samples_in_batch = {num_samples_in_batch}\n")

            print(f"--- Making the Forward Pass for Batch {i} ---")

            # Perform the forward pass
            output, hidden = model(padded_signals_test)

            # Get the prediction labels
            _, predicted = torch.max(output, 1)

            print(f"Pred Values: {predicted}")
            print(f"Labels Eval: {labels_eval}\n")

            # Accumulate the number of correct predictions
            total_correct += (predicted == labels_eval[:predicted.size(0)]).sum().item()

            # Accumulate the number of samples
            total_samples += labels_eval.size(0)

            # Store the predicted and true labels
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels_eval.cpu().numpy())

            # Calculate the probabilities for each class (used by auc and pr_auc below)
            output_probs = torch.softmax(output, dim=1).cpu().numpy()
            all_probabilities.extend(output_probs)

            print(f"---Completed for Batch {i} ---\n")

            if i == 14:
                break

    #  Calculate metrics

    accuracy = total_correct / total_samples
    print(f'\nTest Accuracy: {accuracy}')

    precision = precision_score(all_labels, all_predicted, zero_division=0)
    recall = recall_score(all_labels, all_predicted, zero_division=0)
    f1 = f1_score(all_labels, all_predicted)
    auc = roc_auc_score(all_labels, np.array(all_probabilities)[:, 1], multi_class='ovr')
    pr_auc = average_precision_score(all_labels, np.array(all_probabilities)[:, 1], average='macro')
    conf_matrix = confusion_matrix(all_labels, all_predicted)

    # Metrics for classification performance
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

    # Metric for area under the curve
    print(f'AUC: {auc}')
    print(f'PR AUC: {pr_auc}')

    # Metric to analyze errors the model makes
    print(f'Confusion Matrix:\n{conf_matrix}')



