import matplotlib.pyplot as plt

def plot_training_loss(num_epochs, train_losses):
    # Plotting the loss curves
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    #plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid()
    plt.show()