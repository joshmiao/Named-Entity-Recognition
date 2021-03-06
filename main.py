import torch
import time
import data_process
import model_evaluate
import training

dict_size = 650
data_dir = './data_source/data_source.txt'
device = torch.device('cpu')

st_time = time.time()
data = data_process.preprocess_data(data_dir=data_dir, output_file=None)
dic = data_process.make_dictionary(dict_size=dict_size, data=data)
x_tlist, y_tlist = data_process.create_tensor_list(dic=dic, data=data, device=device)
sample_cnt = len(x_tlist)
test_x_tlist, test_y_tlist = x_tlist[sample_cnt // 31 * 25:], y_tlist[sample_cnt // 31 * 25:]
x_tlist, y_tlist = x_tlist[:sample_cnt // 31 * 25], y_tlist[:sample_cnt // 31 * 25]
print('Pre_processing data done! Used time = {0:} second(s)'.format(time.time() - st_time))

if input('Load theta? (y/n)\n') == 'y':
    theta0 = torch.load('theta0_save.pt')
    theta1 = torch.load('theta1_save.pt')
    theta2 = torch.load('theta2_save.pt')
else:
    theta0 = torch.zeros(3, 3 * (dict_size + 1), device=device, dtype=torch.float32, requires_grad=True)
    theta1 = torch.zeros(3, 3 * (dict_size + 1), device=device, dtype=torch.float32, requires_grad=True)
    theta2 = torch.zeros(3, 3 * (dict_size + 1), device=device, dtype=torch.float32, requires_grad=True)


stop, epoch_tot, training_mode = 0, 0, 0
epoch, learning_rate0, learning_rate1, learning_rate2 = 0, 0, 0, 0
cycle = 1
while stop != 1:
    if input('Input training mode (manual / auto) : ') == 'auto':
        training_mode = 0
    else:
        training_mode = 1
    print('Please input number of epoch :', '(now : {0:})'.format(epoch))
    epoch = eval(input())
    print('Please input learning rate for theta 0 :', '(now : {0:})'.format(learning_rate0))
    learning_rate0 = eval(input())
    print('Please input learning rate for theta 1 :', '(now : {0:})'.format(learning_rate1))
    learning_rate1 = eval(input())
    print('Please input learning rate for theta 2 :', '(now : {0:})'.format(learning_rate2))
    learning_rate2 = eval(input())
    if training_mode == 0:
        f0 = training.auto_grad_training(x_tlist=x_tlist, y_tlist=y_tlist, theta=theta0, theta_num=0,
                                         dict_size=dict_size, epoch=epoch, learning_rate=learning_rate0, device=device)
        f1 = training.auto_grad_training(x_tlist=x_tlist, y_tlist=y_tlist, theta=theta1, theta_num=1,
                                         dict_size=dict_size, epoch=epoch, learning_rate=learning_rate1, device=device)
        f2 = training.auto_grad_training(x_tlist=x_tlist, y_tlist=y_tlist, theta=theta2, theta_num=2,
                                         dict_size=dict_size, epoch=epoch, learning_rate=learning_rate2, device=device)
    else:
        f0 = training.manual_grad_training(x_tlist=x_tlist, y_tlist=y_tlist, theta=theta0, theta_num=0,
                                           dict_size=dict_size, epoch=epoch, learning_rate=learning_rate0,
                                           device=device)
        f1 = training.manual_grad_training(x_tlist=x_tlist, y_tlist=y_tlist, theta=theta1, theta_num=1,
                                           dict_size=dict_size, epoch=epoch, learning_rate=learning_rate1,
                                           device=device)
        f2 = training.manual_grad_training(x_tlist=x_tlist, y_tlist=y_tlist, theta=theta2, theta_num=2,
                                           dict_size=dict_size, epoch=epoch, learning_rate=learning_rate2,
                                           device=device)
    f0.savefig('theta0_cycle{0:}.png'.format(cycle), format='png')
    f1.savefig('theta1_cycle{0:}.png'.format(cycle), format='png')
    f2.savefig('theta2_cycle{0:}.png'.format(cycle), format='png')
    print('Want to continue? (y/n)')
    if input() == 'n':
        stop = 1
    else:
        cycle += 1

check_theta0 = open('check_theta0.log', "w")
check_theta1 = open('check_theta1.log', "w")
check_theta2 = open('check_theta2.log', "w")
model_evaluate.evaluate_model(x_tlist=test_x_tlist, y_tlist=test_y_tlist, theta_num=0, theta=theta0,
                              start=0, end=len(test_x_tlist), output_file=check_theta0, output_prob=True, device=device)
model_evaluate.evaluate_model(x_tlist=test_x_tlist, y_tlist=test_y_tlist, theta_num=1, theta=theta1,
                              start=0, end=len(test_x_tlist), output_file=check_theta1, output_prob=True, device=device)
model_evaluate.evaluate_model(x_tlist=test_x_tlist, y_tlist=test_y_tlist, theta_num=2, theta=theta2,
                              start=0, end=len(test_x_tlist), output_file=check_theta2, output_prob=True, device=device)
