#%%
import torch 
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import time 
import torch.utils.data
import cv2

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:",device)

def read_images(path, num_img):
    array = np.zeros((num_img, 256 * 256))
    i = 0
    for img in os.listdir(path):
        if i >= num_img:
            break

        img_path = os.path.join(path, img)
        img = Image.open(img_path)

        # Resize the image to 256x256
        img = img.resize((256, 256))

        data = np.asarray(img, dtype="uint8")
        data = data.flatten()

        # Crop the data to the first 256 * 256 pixels
        data = data[:256 * 256]

        array[i, :] = data
        i += 1
    return array


# Read train negative
train_neg_path = r"C:/Users/metec/OneDrive/Masaüstü/Resnet/train/neg"
train_neg_num_img = 7500
train_neg_array = read_images(train_neg_path, train_neg_num_img)

# Read train positive
train_pos_path = r"C:/Users/metec/OneDrive/Masaüstü/Resnet/train/pos"
train_pos_num_img = 1500
train_pos_array = read_images(train_pos_path, train_pos_num_img)

# Convert the NumPy arrays to PyTorch tensors
x_train_neg_tensor = torch.from_numpy(train_neg_array[:7500, :]).float()
x_train_pos_tensor = torch.from_numpy(train_pos_array[:1500, :]).float()

# Create the target tensors
y_train_neg_tensor = torch.zeros(7500, dtype=torch.long)
y_train_pos_tensor = torch.ones(1500, dtype=torch.long)




#%% concat train
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:",device)

x_train = torch.cat((x_train_neg_tensor, x_train_pos_tensor),0)
y_train = torch.cat((y_train_neg_tensor, y_train_pos_tensor),0)
print("x_train:",x_train.size())
print("y_train:",y_train.size())

#------------------------------------------------------------------------------------------------------
#read test negative 22050
test_neg_path = r"C:/Users/metec/OneDrive/Masaüstü/Resnet/test/neg"
test_neg_num_img = 2500
test_neg_array = read_images(test_neg_path,test_neg_num_img)

x_test_neg_tensor = torch.from_numpy(test_neg_array[:2500,:])
print("x_test_neg_tensor:",x_test_neg_tensor.size())

y_test_neg_tensor = torch.zeros(2500, dtype = torch.long)
print("y_test_neg_tensor:",y_test_neg_tensor.size())

#read test positive 5944

test_pos_path = r"C:/Users/metec/OneDrive/Masaüstü/Resnet/test/pos"
test_pos_num_img = 500
test_pos_array = read_images(test_pos_path,test_pos_num_img)

x_test_pos_tensor = torch.from_numpy(test_pos_array)
print("x_test_pos_tensor:",x_test_pos_tensor.size())

y_test_pos_tensor = torch.ones(test_pos_num_img, dtype = torch.long)
print("y_test_pos_tensor:",y_test_pos_tensor.size())

#%% concat train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:",device)

x_test = torch.cat((x_test_neg_tensor, x_test_pos_tensor),0)
y_test = torch.cat((y_test_neg_tensor, y_test_pos_tensor),0)
print("x_test:",x_test.size())
print("y_test:",y_test.size())

#%% Visualize 
plt.imshow(x_train[3500,:].reshape(256,256), cmap="gray")

# %% DNN Model

# Hyperparameter

num_epochs = 88
num_classes = 2
batch_size = 32
learning_rate = 0.00001
momentum_ = 0.9

input_size = 65536


# Verilerin GPU'ya taşınması
x_train = x_train.to(device)
y_train = y_train.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)


train = torch.utils.data.TensorDataset(x_train, y_train)
trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

test = torch.utils.data.TensorDataset(x_test, y_test)
testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

# %% Dnn Model

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.35)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x  # shortcut ResNet Model

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)                
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion))
    
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
    
        return nn.Sequential(*layers)
               

    def forward(self, x):
        
        x = x.view(x.size(0), 1, 256, 256)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return(x)

   
        
model = ResNet(BasicBlock, [2, 2, 2], num_classes=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
               
#model = ResNet(BasicBlock,[2, 2, 2]).to(device) if use GPU  

#%% Loss and Optimizer 

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(),lr = learning_rate, momentum=momentum_)      


#%% Train 

start = time.time()

# Train

start = time.time()

loss_list = []
train_acc = []
test_acc = []
use_gpu = True

total_step = len(trainloader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(trainloader):
        # Giriş verilerinizi GPU'ya taşıyın
        images = images.to(device,dtype=torch.float32)

        outputs = model(images)

        loss = criterion(outputs, labels)

        # loss değerini kaydetme
        loss_list.append(loss.item())

        # backward and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 2 == 0:
            print("epoch: {} {}/{}".format(epoch,i,total_step))

    # train
    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            images = images.to(device,dtype=torch.float32)

            outputs = model(images)
            _, predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy train %d %%"%(100*correct/total))
    train_acc.append(100*correct/total)

    # test
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device,dtype=torch.float32)

            outputs = model(images)
            _, predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy test %d %%"%(100*correct/total))
    test_acc.append(100*correct/total)

print("Train is Done !")

         
        
end = time.time()
process_time = (end - start)/60
print("process time:",process_time)

        
#%% Saving Model

torch.save(model.state_dict(), 'trained_DRN_model_Covid3.pth') 
print("Trained model saved.")

#%% Model Visualize        

fig, ax1 = plt.subplots()

plt.plot(loss_list,label = "Loss",color = "black")

ax2 = ax1.twinx()

ax2.plot(np.array(train_acc)/100,label = "Test Acc",color="PURPLE")
ax2.plot(np.array(test_acc)/100,label = "Train Acc",color= "red")
ax1.legend()
ax2.legend()
ax1.set_xlabel('Epoch')
fig.tight_layout()
plt.title("Loss vs Test Accuracy")
plt.show()




#save figure 
plt.savefig('resultus_LSIFIR_DRN.PNG')
print("Training Results Saved.")
#%% Load Model if You Want 

# Boş bir model oluşturun (aynı mimariye sahip)
loaded_model = ResNet()  # YourModelClass, orijinal modelinizi temsil eden sınıf olmalıdır.

# Modelin kaydedilmiş ağırlıklarını yükleyin
loaded_model.load_state_dict(torch.load('model.pth'))

# Modeli eğitim modundan çıkarın (eval moduna geçirin)
loaded_model.eval()


#%%
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

# Loss grafiği
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_list, label='Training Loss', linewidth=1.0)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

# Doğruluk grafiği
plt.subplot(1, 2, 2)
plt.plot(train_acc, label='Training Accuracy', linewidth=1.0)
plt.plot(test_acc, label='Test Accuracy', linewidth=1.0, color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Test Accuracy')
plt.legend()

plt.tight_layout()
plt.show()









        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        






























