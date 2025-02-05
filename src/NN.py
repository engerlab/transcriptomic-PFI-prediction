import torch
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
import torch.utils.data as data_utils


class nn(torch.nn.Module):
    def __init__(self, dimension, dimension2, p):
        super().__init__()

        self.compute = torch.nn.Sequential(
            torch.nn.Linear(dimension, dimension2),
            torch.nn.ReLU(),
            torch.nn.Dropout(p),
            torch.nn.Linear(dimension2, 1),
            torch.nn.ReLU(),
            torch.nn.Dropout(p)
        )

        self.double()
 
    def forward(self, x):
        risk_score = self.compute(x)
        
        return risk_score
        

def loss_function(output, target):
    censorship = target[:,0]
    time = target[:,1]

    # minimize loss so try to get concordance_index as close to 1 as possible
    return 1 - concordance_index_censored(censorship==1, time, output)[0]



def nn_train(model, optimizer, train_set, survival_train, batch_size=50):
    print(type(survival_train[0]))
    print(survival_train)
    
    train_dataset = data_utils.TensorDataset(torch.from_numpy(train_set), torch.from_numpy(survival_train))
    train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):     
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output.cpu().detach().numpy().flatten(), target.cpu().detach().numpy())
        loss.to(device)
        loss.backward()
        optimizer.step()
    return loss.item()




    
