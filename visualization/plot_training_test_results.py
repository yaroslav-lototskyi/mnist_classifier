import matplotlib.pyplot as plt

def plot_training_test_results(train_losses, test_accuracies):
    epochs = range(1, len(train_losses) + 1)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss", color="tab:red")
    ax1.plot(epochs, train_losses, label="Training Loss", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color="tab:blue")
    ax2.plot(epochs, test_accuracies, label="Test Accuracy", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.legend(loc="upper right")

    plt.title("Training Loss and Test Accuracy")
    plt.show()