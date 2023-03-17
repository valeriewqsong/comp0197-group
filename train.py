# copied from my cw1 task2
import torch
import torch.nn as nn
import torch.optim as optim

from mixup_class import mixup
from network import Net

def train_mixup(trainloader, testloader, sampling_method, device):
    # Print the device being used
    print(f"Using device for training: {device}")
    
    accuracy_log = []
    
    # Define parameters
    num_epochs = 2
    alpha = 1.0
    
    mixup_transform = mixup(alpha=alpha, sampling_method=sampling_method)
    
    # Initialize the neural network
    net = Net().to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    # Train the neural network
    print(f'Starting training with Mixup sampling method {sampling_method}...')
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            
            # apply the mixup transformation
            inputs_mix, labels_mix = mixup_transform(inputs, labels)
            inputs_mix, labels_mix = inputs_mix.to(device), labels_mix.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(inputs_mix)
            loss = criterion(outputs, labels_mix)
            loss.backward()
            optimizer.step()
            
            # print statistics every 100 mini batches
            running_loss += loss.item()
            if i % 100 == 99:
                print(f"Epoch {epoch+1}, iteration {i+1}: loss = {running_loss / 100:.5f}")
                running_loss = 0.0
                
        # Calculate test set accuracy and append to history
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                
                # move the images and labels to GPU if available
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                
                # convert the outputs and labels to CPU tensors
                correct += (predicted.cpu() == labels.cpu()).sum().item()
                total += labels.size(0)
        
        # Report the test set performance in terms of classification accuracy versus the epochs. 
        accuracy = 100 * correct / total
        print(f"Accuracy on test set after epoch {epoch+1}: {accuracy:.2f}%")
        accuracy_log.append(accuracy)
        
    print(f'Training with Mixup sampling method {sampling_method} done.')
    
    # save trained model
    torch.save(net.state_dict(), f'saved_model_{sampling_method}.pt')
    print(f'Model with Mixup sampling method {sampling_method} saved.')
    
    return accuracy_log