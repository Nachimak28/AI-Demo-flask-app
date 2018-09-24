from flask import Flask, request, render_template
from werkzeug import secure_filename
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os.path as osp
import numpy as np
from PIL import Image
import os

import glob

files = glob.glob('C:/Users/nachiket/Desktop/RAYD8 tech/Web Apps/uploads/*')
for f in files:
    os.remove(f)

app = Flask(__name__)

class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

model = SmallNet().cpu()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

checkpoint = torch.load('C:/Users/nachiket/Desktop/RAYD8 tech/mnist-4690.pth')
states_to_load = {}
for name, param in checkpoint['state_dict'].items():
    if name.startswith('conv'):
        states_to_load[name] = param

# Construct a new state dict in which the layers we want
# to import from the checkpoint is update with the parameters
# from the checkpoint
model_state = model.state_dict()
model_state.update(states_to_load)
        
model.load_state_dict(model_state)
preprocess = transforms.Compose([transforms.ToTensor()])

model.cpu()  #has to be brought to cpu or else gives some cuda error of float type weights
def image_loader(image_name):
    model.eval()
    image = Image.open(image_name)
    image = preprocess(image).unsqueeze(0)
    val = model(image)
    predicted = torch.max(val, 1)
    CI = np.exp(torch.max(val).numpy()) * 100
    return predicted[1].data.numpy(), CI

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict", methods = ['POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        #pth = "C:\\Users\\nachiket\\Desktop\\RAYD8 tech\\Web Apps\\uploads\\" + str(f.filename)
        a,c  = image_loader(file_path)
        a = str(a)
        c = str(c)
        b = a[1]
        return render_template('result.html', fn = f.filename, pt = "XYZ", text = b, ci = c)
    #f = request.files['file']
    #f.save(secure_filename(f.filename))
    #a = image_loader(f.filename)
    #a = str(a)
    #return a
    #return render_template('index.html', text = a)

if __name__ == "__main__":
    app.run(debug=True)