import torch
import time
import os
import data_process
import model_evaluate
import autograd_training

dict_size = 600
data_dir = './data_source/data_source.txt'
device = torch.device('cpu')
st_time = time.time()

data = data_process.preprocess_data(data_dir=data_dir)

dic = data_process.make_dictionary(dict_size=dict_size, data=data)

x_tlist, y_tlist = data_process.create_tensor_list(dic=dic, data=data, device=device)

print('Pre_processing data done! Used time = {0:} second(s)'.format(time.time() - st_time))

sample_cnt = len(x_tlist)


print('Load theta? (y/n)')
if input() == 'y':
    theta = torch.load('theta_save.pt')
else:
    theta = torch.zeros(3, 3 * (dict_size + 1), device=device, dtype=torch.float32, requires_grad=True)

theta_num = -1
while theta_num not in [0, 1, 2]:
    print('Input theta to train (0/1/2): ')
    theta_num = eval(input())

stop, epoch_tot = 0, 0
epoch, learning_rate = 0, 0

while stop != 1:
    print('Please input number of epoch:', '(now : {0:})'.format(epoch))
    epoch = eval(input())
    print('Please input learning rate', '(now : {0:})'.format(learning_rate))
    learning_rate = eval(input())
    autograd_training.train(x_tlist=x_tlist, y_tlist=y_tlist, theta=theta, theta_num=theta_num, dict_size=dict_size,
                            epoch=epoch, learning_rate=learning_rate, device=device)
    print('Want to continue? (y/n)')
    if input() == 'n':
        stop = 1
