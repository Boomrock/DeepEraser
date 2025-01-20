import pandas as pd
import matplotlib.pyplot as plt


def plot_train_loss(filename):
    try:
        df = pd.read_csv(filename)
        required_columns = ["epoch", "loss"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError("File should contain 'epoch' and 'loss' columns")

        plt.figure(figsize=(10, 6))
        plt.plot(df["epoch"], df["loss"], marker="o")

        plt.title("Training Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    plot_train_loss("./metadata.csv")
