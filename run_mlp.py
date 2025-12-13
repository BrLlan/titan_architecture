import torch
from Basic_MLP import SimpleMLP
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate dataset
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)



input_size = 2
hidden_size = 4
output_size = 1
model = SimpleMLP(input_size, hidden_size, output_size)

#Train  model and store the losses
losses = model.train(X_train, y_train, epochs=1000, lr=0.1)

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.show()

with torch.no_grad():
    test_output = model.forward(X_test)
    test_output = (test_output > 0.5).float() 
accuracy = torch.mean((test_output == y_test).float())
print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")