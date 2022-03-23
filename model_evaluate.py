import torch


def evaluate_model(x_tlist, y_tlist, start, end, theta, theta_num, output_file=None, output_prob=False):
    item_cnt, predict_cnt, true_predict_cnt = 0, 0, 0
    for idx in range(start, end):
        y_prob = [0] * 3
        sigma = 0
        for i in range(3):
            sigma += torch.exp(theta[i] @ x_tlist[idx]).item()
        for i in range(3):
            y_prob[i] = torch.exp(theta[i] @ x_tlist[idx]).item() / sigma
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
    return precision_rate, recall_rate, f1_measure


'''
testing_num = 0
testing_results = open("testing_results{0:}.log".format(testing_num), "w")
print(theta, file=testing_results)
evaluate_model(output_file=testing_results, output_prob=True)

stop = 0
while stop == 0:
    print('continue?(y/n)')
    if input() != 'n':
        testing_num += 1
        print('Input theta name :')
        theta = torch.load(input())
        print('Input theta num :')
        theta_num = eval(input())
        testing_results = open("testing_results{0:}.log".format(testing_num), "w")
        print(theta, file=testing_results)
        evaluate_model(output_file=testing_results, output_prob=True)
    else:
        stop = 1
'''
