#5 numbers -> 10 -> 50 tensors

#50 -> 32 -> 10


import pandas as pd
file = 'train.csv'
df = pd.read_csv(file)

X = df.drop('target', axis=1)
y = df['target']


X = X.values #np data
y = y.values 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41) #32 next

import torch
from torch import nn
import torch.nn.functional as F
class Model(nn.Module):

    def __init__(self, in_features = 50, h1=32, h2=10 ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)

    
    def forward(self, x):
        x = F.relu(self.fc1(x)) #forward propogation -> relu -> Z returning y_pred
        x = self.fc2(x)
        return x 
    


X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

model = Model()

losses = []
EPOCHS = 100

criterion = nn.CrossEntropyLoss()
optimizers = torch.optim.Adam(model.parameters(), lr = 0.01)
for epoch in range(EPOCHS):
    y_pred = model.forward(X_train) #forward propogation
    loss = criterion(y_pred, y_train) #find loss
    losses.append(loss.detach().numpy())
    if epoch%10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss}")
    optimizers.zero_grad()
    loss.backward() #backward propogation
    optimizers.step()




correct = 0

test_file = 'test.csv'
df_test = pd.read_csv(test_file)


X_testing = df_test.drop('id', axis=1)

X_testing = X_testing.values

X_testing = torch.FloatTensor(X_testing)




with torch.no_grad() and open('sample_submission.csv', 'w') as f:
    f.write("id,target\n")
    
    for i, num in enumerate(X_testing):
        y_pred = model.forward(num)
        y_pred = y_pred.argmax().item()
        f.write(f"{i},{y_pred}")
        f.write("\n")



    