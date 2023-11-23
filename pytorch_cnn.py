# Define the class ConvNet which inherits from nn.Module
class ConvNet(nn.Module):
 # Define the constructor of the class
 def __init__(self, num_classes = 4):
  # Call the constructor of the parent class
  super(ConvNet, self).__init__()

  # Define the first convolutional layer
  self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 12, kernel_size=3, stride = 1, padding = 1)
  # Define the batch normalization layer
  self.bn1 = nn.BatchNorm2d(num_features = 12)
  # Define the ReLU activation function
  self.relu1 = nn.ReLU()
  # Define the max pooling layer
  self.pool = nn.MaxPool2d(kernel_size = 2)

  # Define the second convolutional layer
  self.conv2 = nn.Conv2d(in_channels = 12, out_channels = 20, kernel_size = 3, stride = 1, padding = 1)
  # Define the ReLU activation function
  self.relu2 = nn.ReLU()

  # Define the third convolutional layer
  self.conv3 = nn.Conv2d(in_channels = 20, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
  # Define the batch normalization layer
  self.bn3 = nn.BatchNorm2d(num_features=32)
  # Define the ReLU activation function
  self.relu3 = nn.ReLU()

  # Define the first fully connected layer
  self.fc1 = nn.Linear(in_features = 112*112*32, out_features = 120)
  # Define the second fully connected layer
  self.fc2 = nn.Linear(in_features = 120, out_features = num_classes)

 # Define the forward pass
 def forward(self, input):
  # Apply the first convolutional layer
  output = self.conv1(input)
  # Apply the batch normalization layer
  output = self.bn1(output)
  # Apply the ReLU activation function
  output = self.relu1(output)
  # Apply the max pooling layer
  output = self.pool(output)
  # Apply the second convolutional layer
  output = self.conv2(output)
  # Apply the ReLU activation function
  output = self.relu2(output)
  # Apply the third convolutional layer
  output = self.conv3(output)
  # Apply the batch normalization layer
  output = self.bn3(output)
  # Apply the ReLU activation function
  output = self.relu3(output)
  # Reshape the output to a 2D tensor
  output = output.view(-1, 112*112*32)
  # Apply the first fully connected layer
  output = F.relu(self.fc1(output))
  # Apply the second fully connected layer
  output = self.fc2(output)
  # Return the output
  return output
