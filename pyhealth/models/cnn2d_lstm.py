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


# Copyright (c) 2022, Kwanhyung Lee, AITRICS. All rights reserved.
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#  Package that allows you to specify the expected data types of variables, function arguments, and return values
import typing

import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import importlib

# *************************************************************************************************
# ***  Note:  This version of the CNN2D_LTSM model assumes that all data is in the raw format,  ***
# ***         with no feature extraction.
# *************************************************************************************************

class CNN2D_LSTM_V8(nn.Module):

        """
        This class implements a 2D Convolutional Neural Network (CNN) combined with a Long Short-Term Memory (LSTM)
        network for EEG signal processing.  The model is designed to accept raw EEG data as input and perform
        feature extraction using CNN layers, followed by sequence learning with LSTM layers, and finally classification
        using linear layers.

        Args:
            args (typing.Any): An object containing hyperparameters and configuration settings for the model.
                                 It is expected to have attributes such as `num_layers`, `batch_size`, `dropout`,
                                 `num_channel`, and `output_dim`.
            device (torch.device): The device (CPU or CUDA) on which the model will be trained and run.
        """
        def __init__(self, args: typing.Any, device: torch.device):
                super(CNN2D_LSTM_V8, self).__init__()      
                self.args = args
                self.num_layers = args.num_layers
                self.hidden_dim = 256
                self.dropout = args.dropout
                self.num_data_channel = args.num_channel

                #  Note:   Activation is set to 'relu' here.  However, you can choose among different options
                #          for the activation layer if preferred

                activation = 'relu'
                self.activations = nn.ModuleDict([
                        ['lrelu', nn.LeakyReLU()],
                        ['prelu', nn.PReLU()],
                        ['relu', nn.ReLU(inplace=True)],
                        ['tanh', nn.Tanh()],
                        ['sigmoid', nn.Sigmoid()],
                        ['leaky_relu', nn.LeakyReLU(0.2)],
                        ['elu', nn.ELU()]
                ])

                # Create a new variable for the hidden state, necessary to calculate the gradients
                # This is a tuple the contains two tensors (hidden state and cell state).
                self.hidden = ((torch.zeros(self.num_layers, args.batch_size, self.hidden_dim).to(device), torch.zeros(self.num_layers, args.batch_size, self.hidden_dim).to(device)))

                def conv2d_bn(inp, oup, kernel_size, stride, padding):
                        """
                        Helper function to create a 2D convolutional layer followed by batch normalization,
                        activation, and dropout.

                        Args:
                            inp (int): Number of input channels.
                            oup (int): Number of output channels.
                            kernel_size (tuple): Kernel size for the convolution.
                            stride (tuple): Stride for the convolution.
                            padding (tuple): Padding for the convolution.

                        Returns:
                            nn.Sequential: A sequential container of layers.
                        """
                        return nn.Sequential(
                                nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding=padding),
                                nn.BatchNorm2d(oup),
                                self.activations[activation],
                                nn.Dropout(self.dropout),
                )

                # Defines the convolutional feature extraction part of the model
                self.features = nn.Sequential(
                                conv2d_bn(self.num_data_channel,  64, (1,51), (1,4), (0,25)), 
                                nn.MaxPool2d(kernel_size=(1,4), stride=(1,4)),
                                conv2d_bn(64, 128, (1,21), (1,2), (0,10)),
                                conv2d_bn(128, 256, (1,9), (1,2), (0,4)),
                        )

                # Defines an adaptive average pooling layer, pooling input to a 1x1 output size regardless of
                # input feature map size -- useful for getting a fixed-size output before the LSTM
                self.agvpool = nn.AdaptiveAvgPool2d((1,1))

                # Define the LSTM layer
                self.lstm = nn.LSTM(
                        input_size=256,
                        hidden_size=self.hidden_dim,
                        num_layers=args.num_layers,
                        batch_first=True,
                        dropout=args.dropout) 

                # Define the classifier part o the model after LSTM
                self.classifier = nn.Sequential(
                        nn.Linear(in_features=self.hidden_dim, out_features= 64, bias=True),
                        nn.BatchNorm1d(64),
                        self.activations[activation],
                        nn.Linear(in_features=64, out_features= args.output_dim, bias=True),
                )
        
        def forward(self, x):
                """
                Forward pass of the CNN2D_LSTM_V8 model.

                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, num_samples, num_channels).
                                      Note that the input is permuted within the method to (batch_size, 1, num_samples, num_channels)
                                      to fit the CNN2D input requirements.

                Returns:
                    tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]: A tuple containing the output tensor
                                                                           and the LSTM's hidden state.  The output tensor has the shape
                                                                           of (batch_size, output_dim).  The hidden state is a tuple
                                                                           containing the hidden state and cell state, each with shape
                                                                           (num_layers, batch_size, hidden_dim).
                """

                # Extract the batch size from x
                batch_size = x.size(0)

                # Initializes the LSTM's hidden state
                hidden = self.init_state(x.device, batch_size)

                # Rearrange x to match the expected input format of the layers
                x = x.permute(0, 2, 1)

                # Adds an extra dimension to x to represent the channel dimension for 2D convolutions
                x = x.unsqueeze(1)

                # Passes x through convolutional feature extraction layers
                x = self.features(x)

                # Apply adaptive average pooling
                x = self.agvpool(x)

                # Remove the extra dimension and rearrange the dimension to fit the LSTM input requirements
                x = torch.squeeze(x, 2)
                x = x.permute(0, 2, 1)

                # Note:  This line of code was in the original model.  It's commented out because we
                # initiative the hidden state above
                # self.hidden = tuple(([Variable(var.data) for var in self.hidden]))

                # Pass the data through the LSTM layer
                output, hidden = self.lstm(x, hidden)

                # Select the output of the LSTM at the last time step
                output = output[:,-1,:]

                # Pass the LSTM output through the classifier (linear layers)
                output = self.classifier(output)

                return output, hidden

                # This line of code was commented out in the original model
                # return torch.sigmoid(output), self.hidden
                

        def init_state(self, device, batch_size):
                """
                Initializes the LSTM's hidden state to zero-filled tensors.

                Args:
                    device (torch.device): The device (CPU or CUDA) where the hidden state should be stored.
                    batch_size (int): The batch size of the input data.

                Returns:
                    tuple[torch.Tensor, torch.Tensor]: A tuple containing the initialized hidden state
                                                     and cell state, each with shape
                                                     (num_layers, batch_size, hidden_dim).
                """
                hidden = ((torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
                           torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)))
                return hidden

                # These lines of code were commented out in the original model
                # hs_forward = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
                # cs_forward = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
                # return (torch.nn.init.kaiming_normal_(hs_forward),
                #         torch.nn.init.kaiming_normal_(cs_forward))


