import torch


class SimpleMLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = torch.randn(input_size, hidden_size, requires_grad=True)
        self.b1 = torch.randn(1, hidden_size, requires_grad=True)
        self.W2 = torch.randn(hidden_size, output_size, requires_grad=True)
        self.b2 = torch.randn(1, output_size, requires_grad=True)

    def forward(self, X):
        self.z1 = torch.matmul(X, self.W1) + self.b1
        self.a1 = torch.sigmoid(self.z1)  # Hidden layer activation
        self.z2 = torch.matmul(self.a1, self.W2) + self.b2
        self.a2 = torch.sigmoid(self.z2)  # Output layer activation
        return self.a2

    def backward(self,X,y,output,lr=0.01):
        m=X.shape[0]
        dz2=output-y
        dW2=torch.matmul(self.a1.T,dz2)
        db2=torch.sum(dz2,axis=0)/m

        da1=torch.matmul(dz2,self.W2.T)
        dz1=da1*(self.a1*(1-self.a1))
        dw1=torch.matmul(X.T,dz1)/m
        db1 = torch.sum(dz1, axis=0) / m
        
        with torch.no_grad():
            self.W1 -= lr * dw1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            
    def train(self, X, y, epochs=1000, lr=0.01):
        losses = []
        for epoch in range(epochs):
            output = self.forward(X)
            #Compute loss using (Mean Squared Error)
            loss = torch.mean((output - y) ** 2)
            losses.append(loss.item())
            #update weights
            self.backward(X, y, output, lr)
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        return losses        