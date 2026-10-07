import matplotlib.pyplot as plt
from pathlib import Path

epochs = list(range(1, 9))

train_loss = [5.3483, 4.7961, 4.5189, 4.3751, 4.2812, 4.2071, 4.1445, 4.0902]
val_loss = [4.9718, 4.5741, 4.4227, 4.3459, 4.3013, 4.2755, 4.2610, 4.2490]

train_accuracy = [0.0213, 0.0658, 0.0919, 0.1026, 0.1108, 0.1178, 0.1246, 0.1300]
val_accuracy = [0.0434, 0.0870, 0.1018, 0.1075, 0.1145, 0.1185, 0.1219, 0.1244]

Path("output").mkdir(exist_ok=True)

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, marker="o", label="Training Loss")
plt.plot(epochs, val_loss, marker="o", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("LSTM Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("output/loss_curve.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_accuracy, marker="o", label="Training Accuracy")
plt.plot(epochs, val_accuracy, marker="o", label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("LSTM Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("output/accuracy_curve.png", dpi=300)
plt.close()

print("Training graphs created successfully!")
print("Saved:")
print("output/loss_curve.png")
print("output/accuracy_curve.png")
