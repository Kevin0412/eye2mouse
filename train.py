import torch
import tqdm
import numpy as np

torch.manual_seed(1)    # reproducible

file=open('data.csv','r',encoding='utf-8')
file1=file.read()
file.close()
data=file1.split('\n')
x_data=[]
y_data=[]
for d in tqdm.tqdm(data):
    if d!=data[0] and d!='':
        d=d.split(',')
        x_data.append([])
        y_data.append([])
        for n in range(len(d)):
            if n==0:
                y_data[-1].append(float(d[n])/1920)
            elif n==1:
                y_data[-1].append(float(d[n])/1010)
            else:
                x_data[-1].append(float(d[n]))

x_data=np.array(x_data,np.float32)
y_data=np.array(y_data,np.float32)
x = torch.unsqueeze(torch.from_numpy(x_data),dim=1)  # x data (tensor), shape=(2290, 63)
y = torch.unsqueeze(torch.from_numpy(y_data),dim=1)  # y data (tensor), shape=(2290, 2)

# torch can only train on Variable, so convert them to Variable
# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


net=torch.nn.Sequential(
    torch.nn.Linear(63,64),
    torch.nn.ReLU(),
    torch.nn.Linear(64,64),
    torch.nn.ReLU(),
    torch.nn.Linear(64,2),
)    # define the network
print(net)  # net architecture

optimizer = torch.optim.Adam(net.parameters(), lr=0.01,betas=(0.9,0.99))
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
patience=10000
prediction = net(x)     # input x and predict based on x
min_loss = loss_func(prediction, y).data.numpy()     # must be (1. nn output, 2. target)

backward_times=0
file=open('train.csv','w',encoding='utf-8')
file.write('times,loss\n')
t=0
while True:
    prediction = net(x)     # input x and predict based on x

    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if loss.data.numpy()<min_loss:
        torch.save(net,'best.pkl')
        backward_times=0
        min_loss=loss.data.numpy()
    else:
        backward_times+=1

    file.write('{},{}\n'.format(t+1,loss.data.numpy()))

    print('\rtimes:{}    loss:{}    backward_times:{}   '.format(t+1,loss.data.numpy(),backward_times),end='')
    
    if backward_times>patience:
        break

    t+=1

torch.save(net,'last.pkl')
file.close()