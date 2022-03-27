import torch
import time


def evaluate_model(x_tlist, y_tlist, start, end, theta, theta_num, device, output_file=None, output_prob=False):
    st_time = time.time()
    item_cnt, predict_cnt, true_predict_cnt = 0, 0, 0
    loss = torch.tensor(0, dtype=torch.float32, device=device)
    for idx in range(start, end):
        y_prob = []
        sigma = torch.tensor(1, dtype=torch.float32, device=device)
        for i in range(2):
            sigma += torch.exp(theta[i] @ x_tlist[idx])
        for i in range(2):
            y_prob.append(torch.exp(theta[i] @ x_tlist[idx]) / sigma)
        y_prob.append(1 / sigma)
        loss -= torch.log(y_prob[y_tlist[idx][theta_num]])
        y_predict, max_prob = 0, 0
        for i in range(3):
            if y_prob[i] > max_prob:
                max_prob = y_prob[i]
                y_predict = i
        if y_predict != 0:
            predict_cnt += 1
        if y_tlist[idx][theta_num].item() != 0:
            item_cnt += 1
            if y_predict == y_tlist[idx][theta_num].item():
                true_predict_cnt += 1
        if output_prob:
            print(y_prob, y_tlist[idx][theta_num].item(), file=output_file)
    print('Evaluation For theta {0:} (Used time = {1:} second(s)) : '.format(theta_num, time.time() - st_time),
          file=output_file)
    loss /= (end - start)
    print('loss =', loss.item())
    print('item_cnt =', item_cnt, '| predict_cnt =', predict_cnt, '| true_predict_cnt =', true_predict_cnt,
          file=output_file)
    precision_rate, recall_rate, f1_measure = 0, 0, 0
    if item_cnt != 0:
        recall_rate = true_predict_cnt / item_cnt
    if predict_cnt != 0:
        precision_rate = true_predict_cnt / predict_cnt
    if precision_rate + recall_rate != 0:
        f1_measure = 2 * precision_rate * recall_rate / (precision_rate + recall_rate)
    print('precision_rate =', precision_rate, '| recall_rate =', recall_rate,
          '| F1_measure =', f1_measure,
          file=output_file)
    return loss.item(), precision_rate, recall_rate, f1_measure
