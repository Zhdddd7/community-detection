import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# 生成示例数据
ratings = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4]
])

class RatingDataset(Dataset):
    def __init__(self, ratings):
        self.data = []
        self.labels = []
        for user_id in range(ratings.shape[0]):
            for item_id in range(ratings.shape[1]):
                if ratings[user_id, item_id] > 0:
                    self.data.append([user_id, item_id])
                    self.labels.append(ratings[user_id, item_id])
        self.data = torch.tensor(self.data, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class FactorizationMachine(nn.Module):
    def __init__(self, num_users, num_items, k):
        super(FactorizationMachine, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.k = k
        
        self.user_embedding = nn.Embedding(num_users, k)
        self.item_embedding = nn.Embedding(num_items, k)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(0.0))
        
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        user_bias = self.user_bias(user).squeeze()
        item_bias = self.item_bias(item).squeeze()
        
        interaction = torch.sum(user_emb * item_emb, dim=1)
        prediction = self.global_bias + user_bias + item_bias + interaction
        
        return prediction

def train_fm(model, dataloader, epochs=10, lr=0.01, dir = "models"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"training FM model, running on {device}")
    criterion = nn.MSELoss().to(device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, labels in dataloader:
            user, item = data[:, 0].to(device), data[:, 1].to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            predictions = model(user, item)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")
        if (epoch+1) % 100 ==0:
            torch.save(model, f"{dir}/FM_{epoch}_epoch.pt")
            print(f"{epoch} epoch model is saved.")

def predict_all(model, num_users, num_items):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    predictions = np.zeros((num_users, num_items))
    with torch.no_grad():
        for user in range(num_users):
            for item in range(num_items):
                user_tensor = torch.tensor([user], dtype=torch.long).to(device)
                item_tensor = torch.tensor([item], dtype=torch.long).to(device)
                prediction = model(user_tensor, item_tensor).item()
                predictions[user, item] = prediction
    return predictions

def matrixFill(ratings, train = True):
    dataset = RatingDataset(ratings)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    num_users = ratings.shape[0]
    num_items = ratings.shape[1]
    k = 10  # hidden dimension 
    if train:
        fm_model = FactorizationMachine(num_users, num_items, k)
        train_fm(fm_model, dataloader, epochs=300, lr=0.01)
    else:
        dir = "models"
        fm_model = torch.load(f"{dir}/FM_299_epoch.pt")
    
    completed_matrix = predict_all(fm_model, num_users, num_items)
    completed_matrix[completed_matrix <0] =0
    completed_matrix[completed_matrix >5] = 5
    print("矩阵已补全")
    return completed_matrix


