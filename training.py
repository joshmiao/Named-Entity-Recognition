import torch
import time
import os
import model_evaluate
import matplotlib.pyplot as plt


def auto_grad_training(x_tlist, y_tlist, theta, theta_num, dict_size, epoch, learning_rate, device):
    if len(x_tlist) != len(y_tlist):
        print('Data error!')
    seq, v_loss_list, t_loss_list, f1_list = [], [], [], []
    training_cnt = len(x_tlist) // 5 * 4
    print('Initial theta {:0}: '.format(theta_num))
    print(theta)
    print('-------------------------------------------------------------------------------------------------------')
    for __epoch__idx__ in range(epoch):
        print("epoch {0:} for theta {1:} ".format(__epoch__idx__, theta_num), end='')
        st_time = time.time()
        # using softmax model to give probability and using entropy loss to optimize theta #
        # noting that the last dimension of theta is zero #
        loss = torch.tensor(0, device=device, dtype=torch.float32)
        for idx in range(training_cnt):
            sigma = torch.tensor(1, device=device, dtype=torch.float32)
            for i in range(2):
                sigma += torch.exp(theta[i] @ x_tlist[idx])
            loss += torch.log(sigma)
            if y_tlist[idx][theta_num] != 2:
                loss -= theta[y_tlist[idx][theta_num]] @ x_tlist[idx]
        loss /= training_cnt
        # calculate grad using backward method
        loss.backward(gradient=torch.tensor(1, dtype=torch.float32, device=device))
        with torch.no_grad():
            theta -= learning_rate * theta.grad
            theta.grad = None
        print('(Used time = {0:} second(s)) : '.format(time.time() - st_time))
        print('training loss =', loss.item())
        print(theta)
        validation_loss, precision_rate, recall_rate, f1_measure = model_evaluate.evaluate_model(
                                                                                x_tlist=x_tlist, y_tlist=y_tlist,
                                                                                start=training_cnt, end=len(x_tlist),
                                                                                theta=theta, theta_num=theta_num,
                                                                                device=device)
        v_loss_list.append(validation_loss)
        t_loss_list.append(loss.item())
        f1_list.append(f1_measure)
        seq.append(__epoch__idx__ + 1)
        print('---------------------------------------------------------------------------------------------------')
        if not os.path.exists('./theta{0:}_save/'.format(theta_num)):
            os.mkdir('./theta{0:}_save/'.format(theta_num))
        torch.save(theta,
                   './theta{0:}_save/theta{1:}_save_tmp_{2:}_dic_size={3:}_F1={4:.4f}_li={5:.4f}.pt'
                   .format(theta_num, theta_num, __epoch__idx__, dict_size, f1_measure, loss.item()))
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    axs[0].set_title('Loss', fontsize=24)
    axs[0].set_xlabel('Number of epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_xticks(seq)
    axs[0].plot(seq, t_loss_list, label='training_loss')
    axs[0].plot(seq, v_loss_list, label='validation_loss')
    axs[0].legend()
    axs[1].plot(seq, f1_list, label='validation_dataset_f1_measure')
    axs[1].set_title('F1_measure', fontsize=24)
    axs[1].set_xlabel('Number of epoch')
    axs[1].set_ylabel('F1_measure')
    axs[1].set_xticks(seq)
    axs[1].legend()
    fig.tight_layout(pad=3, h_pad=1.0)
    return fig


def manual_grad_training(x_tlist, y_tlist, theta, theta_num, dict_size, epoch, learning_rate, device):
    if len(x_tlist) != len(y_tlist):
        print('Data error!')
    seq, v_loss_list, t_loss_list, f1_list = [], [], [], []
    training_cnt = len(x_tlist) // 5 * 4
    print('Initial theta {:0}: '.format(theta_num))
    print(theta)
    print('-------------------------------------------------------------------------------------------------------')
    for __epoch__idx__ in range(epoch):
        print("epoch {0:} for theta {1:} ".format(__epoch__idx__, theta_num), end='')
        st_time = time.time()
        # using softmax model to give probability and using entropy loss to optimize theta #
        # noting that the last dimension of theta is zero #
        loss = torch.tensor(0, device=device, dtype=torch.float32)
        grad = torch.zeros(3, 3 * (dict_size + 1), device=device, dtype=torch.float32)
        for idx in range(training_cnt):
            sigma = torch.tensor(1, device=device, dtype=torch.float32)
            s = []
            for i in range(2):
                s.append(theta[i] @ x_tlist[idx])
                sigma += torch.exp(s[i])
            loss += torch.log(sigma)
            if y_tlist[idx][theta_num] != 2:
                loss -= s[y_tlist[idx][theta_num]]
            for i in range(2):
                if i != y_tlist[idx][theta_num]:
                    grad[i] += torch.exp(s[i]) / sigma * x_tlist[idx]
                else:
                    grad[i] += (torch.exp(s[i]) / sigma - 1) * x_tlist[idx]

        loss /= training_cnt
        grad /= training_cnt
        with torch.no_grad():
            theta -= learning_rate * grad
        print('(Used time = {0:} second(s)) : '.format(time.time() - st_time))
        print('training loss =', loss.item())
        print(theta)
        validation_loss, precision_rate, recall_rate, f1_measure = model_evaluate.evaluate_model(
                                                                                x_tlist=x_tlist, y_tlist=y_tlist,
                                                                                start=training_cnt, end=len(x_tlist),
                                                                                theta=theta, theta_num=theta_num,
                                                                                device=device)
        v_loss_list.append(validation_loss)
        t_loss_list.append(loss.item())
        f1_list.append(f1_measure)
        seq.append(__epoch__idx__ + 1)
        print('---------------------------------------------------------------------------------------------------')
        if not os.path.exists('./theta{0:}_save/'.format(theta_num)):
            os.mkdir('./theta{0:}_save/'.format(theta_num))
        torch.save(theta,
                   './theta{0:}_save/theta{1:}_save_tmp_{2:}_dic_size={3:}_F1={4:.4f}_li={5:.4f}.pt'
                   .format(theta_num, theta_num, __epoch__idx__, dict_size, f1_measure, loss))
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    axs[0].set_title('Loss', fontsize=24)
    axs[0].set_xlabel('Number of epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_xticks(seq)
    axs[0].plot(seq, t_loss_list, label='training_loss')
    axs[0].plot(seq, v_loss_list, label='validation_loss')
    axs[0].legend()
    axs[1].plot(seq, f1_list, label='validation_dataset_f1_measure')
    axs[1].set_xticks(seq)
    axs[1].set_title('F1_measure', fontsize=24)
    axs[1].set_xlabel('Number of epoch')
    axs[1].set_ylabel('F1_measure')
    axs[1].legend()
    fig.tight_layout(pad=3, h_pad=1.0)
    return fig
