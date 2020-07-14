import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import pickle
from dataset import VehicleDataset
from lstm import VehicleLSTM
import matplotlib.pyplot as plt


def main():

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    train_dataset = VehicleDataset(mode='train')
    test_dataset = VehicleDataset(mode='test')
    train_loader = DataLoader(dataset=train_dataset, batch_size=20, shuffle=False, num_workers=4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=20, shuffle=False, num_workers=4)

    # hyper-parameters
    num_epochs = 20
    lr = 0.0005

    model = VehicleLSTM(
        input_size=30, 
        hidden_size=256,
        output_size=2, 
        num_layers=1, 
        dropout=0.1, 
        device=device
    ).to(device)

    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    optim = Adam(model.parameters(), lr=lr)

    train_loss = []
    test_loss = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for n_batch, (in_batch, label) in enumerate(train_loader):
            in_batch, label = in_batch.to(device), label.to(device)
            pred = model(in_batch)

            loss = mse_loss(pred, label)
            epoch_loss += loss.item()

            optim.zero_grad()
            loss.backward()
            optim.step()
        train_loss.append(epoch_loss)

        l1_err, l2_err = 0, 0
        lateral_loss, long_loss = 0, 0
        losses = []
        model.eval()
        with torch.no_grad():
            for n_batch, (in_batch, label) in enumerate(test_loader):
                if n_batch == 1:
                    in_batch, label = in_batch.to(device), label.to(device)
                    pred = model.test(in_batch)
                    if epoch == num_epochs - 1 and n_batch == 1:
                        print('pred: ', pred)
                        print('label: ', label)

                    l1_err += l1_loss(pred, label).item()
                    l2_err += mse_loss(pred, label).item()
                    lateral_loss += mse_loss(pred[:,:,0], label[:,:,0]).item()
                    long_loss += mse_loss(pred[:,:,1], label[:,:,1]).item()

        test_loss.append(l2_err)

            # if (n_batch + 1) % 10 == 0:
            #     print("Epoch: [{}/{}], Batch: {}, Loss: {}".format(
            #         epoch, num_epochs, n_batch, loss.item()))
        print('Epoch: [{}/{}], Loss: {}'.format(epoch, num_epochs, epoch_loss))
        # print('train loss: ', train_loss)
        # print('test loss: ', test_loss)


    
    # l1_err, l2_err = 0, 0
    # l1_loss = nn.L1Loss()
    # lateral_loss, long_loss = 0, 0
    # losses = []
    # model.eval()
    # gt = []
    # l = []
    # with torch.no_grad():
    #     for n_batch, (in_batch, label) in enumerate(test_loader):
    #         in_batch, label = in_batch.to(device), label.to(device)
    #         pred = model.test(in_batch)
    #         if n_batch == 0:
    #             print('pred shape: ', pred.shape)
    #             print('label shape: ', label.shape)

    #         l1_err += l1_loss(pred, label).item()
    #         l2_err += mse_loss(pred, label).item()

    print('Test L1 error:', l1_err)
    print('Test L2 error:', l2_err)
    print('Longitudinal error: ', long_loss)
    print('Lateral error: ', lateral_loss)




    # if device is 'cpu':
    #     pred = pred.detach().numpy()[0,:,:]
    #     label = label.detach().numpy()[0,:,:]
    # else:
    #     pred = pred.detach().cpu().numpy()[0,:,:]
    #     label = label.detach().cpu().numpy()[0,:,:]

    # r = []
    # num_points = 17
    # interval = 1./num_points
    # x = int(num_points/2)
    # for j in range(-x,x+1):
    #     r.append(interval*j)

    # from matplotlib import pyplot as plt
    # plt.figure()
    # for i in range(1, len(pred)):
    #     c = (i/(num_points+1), 1-i/(num_points+1), 0.5)
    #     plt.plot(pred[i], r, label='t = %s' %(i), c=c)
    # plt.xlabel('velocity [m/s]')
    # plt.ylabel('r [m]')
    # plt.legend(bbox_to_anchor=(1,1),fontsize='x-small')
    # plt.show()

    # plt.figure()
    # for i in range(1, len(label)):
    #     c = (i/(num_points+1), 1-i/(num_points+1), 0.5)
    #     plt.plot(label[i], r, label='t = %s' %(i), c=c)
    # plt.xlabel('velocity [m/s]')
    # plt.ylabel('r [m]')
    # plt.legend(bbox_to_anchor=(1,1),fontsize='x-small')
    # plt.show()


if __name__ == "__main__":
    main()