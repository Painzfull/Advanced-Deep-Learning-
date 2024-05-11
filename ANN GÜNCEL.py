
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
# %%
from sklearn.preprocessing import MinMaxScaler
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

# CUDA aktifliğini kontrol etme
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'CUDA aktif mi: {torch.cuda.is_available()}')

# Veri setini yükleme
data = pd.read_csv('C:/Users/Froggremann/Desktop/HASTA-SAĞLIKLI.csv', encoding="utf-8")

# Veri setinin ilk birkaç satırını yazdırma
print("Veri seti:")
print(data.head())

# Veri setinin boyutunu yazdırma
print("\nVeri setinin boyutu:", data.shape)

# Özellikler ve hedef değerlerini ayırma
X = data.drop(columns=['HASTA/DEĞİL'])
y = data['HASTA/DEĞİL']

# Veri setini eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle= True)

# Verileri ölçeklendirme
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Verileri tensorlere dönüştürme
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train.values, dtype=torch.long).to(device)
y_test = torch.tensor(y_test.values, dtype=torch.long).to(device)

# Eğitim ve test setlerinin boyutlarını yazdırma
print("\nEğitim setinin boyutu:", X_train.shape)
print("Test setinin boyutu:", X_test.shape)
#%%
import matplotlib.pyplot as plt

# Hasta ve sağlıklı olanların sayısını hesapla
num_sick_train = (y_train == 1).sum().item()
num_healthy_train = (y_train == 0).sum().item()
num_sick_test = (y_test == 1).sum().item()
num_healthy_test = (y_test == 0).sum().item()

# Histogram çiz
plt.figure(figsize=(12,6))

# Train verileri için histogram
plt.subplot(1, 2, 1)
plt.bar(['Sağlıklı', 'Hasta'], [num_healthy_train, num_sick_train], color=['lightgreen', 'lightcoral'])
plt.ylabel('Kişi Sayısı')
plt.title('Train Veri Seti Dağılımı')
for i, v in enumerate([num_healthy_train, num_sick_train]):
    plt.text(i, v, str(v), color='black', fontweight='bold')

# Test verileri için histogram
plt.subplot(1, 2, 2)
plt.bar(['Sağlıklı', 'Hasta'], [num_healthy_test, num_sick_test], color=['lightgreen', 'lightcoral'])
plt.ylabel('Kişi Sayısı')
plt.title('Test Veri Seti Dağılımı')
for i, v in enumerate([num_healthy_test, num_sick_test]):
    plt.text(i, v, str(v), color='black', fontweight='bold')

plt.show()


# %%
import torch.nn as nn
import torch.nn.functional as F

class ANNClassifier(nn.Module):
    def __init__(self):
        super(ANNClassifier, self).__init__()
        self.fc1 = nn.Linear(9, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.25)  # 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # 
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # 
        x = F.relu(self.fc3(x))
        x = self.dropout(x)  # 
        x = F.log_softmax(self.fc4(x), dim=1)
        return x




# Modeli oluşturma ve CUDA'ya taşıma
model = ANNClassifier().to(device)

# Modelin giriş boyutunu ve çıkış boyutunu kontrol etme
sample_input = torch.randn(1, 9).to(device)
print(f"Giriş boyutu: {sample_input.shape}")
print(f"Çıkış boyutu: {model(sample_input).shape}")


#%%

# Epoch sayısını, batch boyutunu ve öğrenme oranını ayarlama
num_epochs = 100
batch_size = 1
learning_rate = 0.001

# Loss fonksiyonunu ve optimizer'ı tanımlama
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

#%%
from tqdm import tqdm

# Epochlar boyunca eğitim ve test işlemleri
for epoch in range(num_epochs):
    # Eğitim
    model.train()
    running_loss_train = 0.0
    correct_train = 0
    total_train = 0
    for i in tqdm(range(0, len(X_train), batch_size), desc=f'Epoch {epoch+1}/{num_epochs} - Train', unit=' batches'):
        optimizer.zero_grad()
        batch_X_train = X_train[i:i+batch_size]
        batch_y_train = y_train[i:i+batch_size]
        output = model(batch_X_train)
        loss_train = criterion(output, batch_y_train.long())
        loss_train.backward()
        optimizer.step()
        running_loss_train += loss_train.item()
        _, predicted_train = torch.max(output, 1)
        correct_train += int((predicted_train == batch_y_train).sum().item())
        total_train += len(batch_X_train)

    accuracy_train = correct_train / total_train

    # Test
    model.eval()
    running_loss_test = 0.0
    correct_test = 0
    total_test = 0
    for i in tqdm(range(0, len(X_test), batch_size), desc=f'Epoch {epoch+1}/{num_epochs} - Test', unit=' batches'):
        batch_X_test = X_test[i:i+batch_size]
        batch_y_test = y_test[i:i+batch_size]
        output = model(batch_X_test)
        loss_test = criterion(output, batch_y_test.long())
        running_loss_test += loss_test.item()
        _, predicted_test = torch.max(output, 1)
        correct_test += int((predicted_test == batch_y_test).sum().item())
        total_test += len(batch_X_test)

    accuracy_test = correct_test / total_test

    print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {running_loss_train / (len(X_train) / batch_size)}, Test Loss: {running_loss_test / (len(X_test) / batch_size)}, Train Accuracy: {accuracy_train * 100:.2f}%, Test Accuracy: {accuracy_test * 100:.2f}%')
