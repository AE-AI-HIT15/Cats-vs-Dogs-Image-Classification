import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_results(history):
    history_df = pd.DataFrame(history.history)

    plt.figure(figsize=(18, 5), dpi=200)
    sns.set_style('darkgrid')

    # Biểu đồ Cross Entropy Loss
    plt.subplot(121)
    plt.title('Cross Entropy Loss', fontsize=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.plot(history_df['loss'], label='Train Loss')
    plt.plot(history_df['val_loss'], label='Validation Loss')
    plt.legend()

    # Biểu đồ Accuracy
    plt.subplot(122)
    plt.title('Classification Accuracy', fontsize=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.plot(history_df['accuracy'], label='Train Accuracy')
    plt.plot(history_df['val_accuracy'], label='Validation Accuracy')
    plt.legend()

    plt.show()