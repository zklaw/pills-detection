import torch
import numpy as np
from functions import plot_results

def training_loop(num_epochs, model, trainloader, valloader, optimizer, save_path):
    
    val_loss_n = [100000, 99999]
    train_loss_n = [100000, 99999]
    
    for epoch in range(num_epochs):

        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        train_loss = []
        val_loss = []

        model.train()
        for inputs, targets in trainloader:

            inputs = [input.permute(2, 0, 1) for input in inputs] #H W C to C H W
            outputs = model(inputs, targets)
            losses = sum(loss for loss in outputs.values())
            loss_value = losses.item()
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            train_loss.append(loss_value)

        with torch.no_grad():
            for inputs, targets in valloader:
                inputs = [input.permute(2, 0, 1) for input in inputs]
                outputs = model(inputs, targets)
                losses = sum(loss for loss in outputs.values())
                loss_value = losses.item()

                val_loss.append(loss_value)

            train_loss = np.sum(train_loss)/len(train_loss)
            val_loss = np.sum(val_loss)/len(val_loss)

            if val_loss < np.min(val_loss_n):
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': val_loss,
                        }, save_path)
                print('\nmodel saved!\n')

            train_loss_n.append(train_loss)
            val_loss_n.append(val_loss)

        print(f'Training loss: {train_loss:.2f}, Validation loss: {val_loss:.2f}')
        
    return model

def testing_loop(model, testloader, threshold):

    model.eval()

    for inputs, targets in testloader:
        images = []
        for img in inputs:
            images.append(img.cpu().detach().numpy())

        inputs = [input.permute(2, 0, 1) for input in inputs] 

        outputs = model(inputs)

        plot_results(images, outputs, targets, threshold)
        break