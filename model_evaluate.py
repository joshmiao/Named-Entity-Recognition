import torch
import time
import os
import numpy as np
data_source = open('./data_source/data_source.txt', 'r')
st_time = time.time()


'''
the 'data' list contains 'word's in the following format:
    word : [text, type, organization_mark, place_mark, name_mark]
        text : {string}
        type : {string}
        organization_mark : {0, 1, 2} 
        place_mark : {0, 1, 2}
        name_mark : {0, 1, 2}
            0 for O(Outside), 1 for B(Begin), 2 for I(Inside)
'''
data = list()

# processing and marking data

for lines in data_source:
    line_data = lines.split('  ')
    for word in line_data:
        if len(word.split('/')) == 2:
            data.append(word.split('/'))

for word in data:
    mark = []
    mark.append(1) if word[1] == 'nt' else mark.append(0)
    mark.append(1) if word[1] == 'ns' else mark.append(0)
    if word[1] == 'nr':
        mark.append(1)
    elif word[1] == 'nrf':
        mark.append(1)
    elif word[1] == 'nrg':
        mark.append(2)
    else:
        mark.append(0)
    word += mark

idx = 0
while idx < len(data):
    if data[idx][0].find('[') != -1:
        # find the end of a combination name
        end_id = idx + 1
        while data[end_id][1].find(']') == -1:
            end_id += 1
        # process name and type
        ty = data[end_id][1].split(']')[1]
        data[end_id][1] = data[end_id][1].split(']')[0]
        data[idx][0] = data[idx][0].split('[')[1]
        # marking Begin item
        if ty == 'nt':
            data[idx][2] = 1
        elif ty == 'ns':
            data[idx][3] = 1
        elif ty == 'nr':
            data[idx][4] = 1
        idx += 1
        # marking Inside item
        while idx <= end_id:
            if ty == 'nt':
                data[idx][2] = 2
            elif ty == 'ns':
                data[idx][3] = 2
            elif ty == 'nr':
                data[idx][4] = 2
            idx += 1
    else:
        idx += 1

# data_output = open("./data_source/data_output.txt", "w")
# print(data, file=data_output)

# making dictionary of commonly used words

dict_size = 600
cnt = dict()
dic = dict()
cnt_ordered = list()

for word in data:
    cnt[word[0]] = cnt.get(word[0], 0) + 1

for key in cnt:
    cnt_ordered.append([key, cnt[key]])
cnt_ordered.sort(key=lambda a: a[1], reverse=True)
dict_size = min(dict_size, len(cnt_ordered))

for idx in range(dict_size):
    dic[cnt_ordered[idx][0]] = idx + 1

# creating tensor list


device = torch.device('cpu')
# device = torch.device('cuda:0')
x_tlist = list()
y_tlist = list()
print('Total', len(data), 'words')
for idx in range(1, len(data) - 1):
    x = [0] * (dict_size + 1) * 3
    label = [data[idx][2], data[idx][3], data[idx][4]]
    x[dic.get(data[idx - 1][0], 0)] = 1
    x[dic.get(data[idx][0], 0) + (dict_size + 1)] = 1
    x[dic.get(data[idx + 1][0], 0) + 2 * (dict_size + 1)] = 1
    x_tlist.append(torch.tensor(x, device=device, dtype=torch.float))
    y_tlist.append(torch.tensor(label, device=device, dtype=torch.int))

print('Pre_processing data done! Used time = {0:} second(s)'.format(time.time() - st_time))
#################################################################################

if len(x_tlist) != len(y_tlist):
    print('Data error!')

sample_cnt = len(x_tlist)
training_cnt = sample_cnt // 3 * 2
validation_cnt = sample_cnt // 6
testing_cnt = sample_cnt - training_cnt - validation_cnt
print('Input theta name :')
theta = torch.load(input())
print('Input theta num :')
theta_num = eval(input())


def evaluate_model(output_file=None, output_prob=False):
    item_cnt, predict_cnt, true_predict_cnt = 0, 0, 0
    for idx in range(training_cnt + validation_cnt, sample_cnt):
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
    print('item_cnt =', item_cnt, 'predict_cnt =', predict_cnt, 'true_predict_cnt =', true_predict_cnt,
          file=output_file)
    print('item_cnt =', item_cnt, 'predict_cnt =', predict_cnt, 'true_predict_cnt =', true_predict_cnt)
    precision_rate, recall_rate, f1_measure = 0, 0, 0
    if item_cnt != 0:
        recall_rate = true_predict_cnt / item_cnt
    if predict_cnt != 0:
        precision_rate = true_predict_cnt / predict_cnt
    if precision_rate + recall_rate != 0:
        f1_measure = 2 * precision_rate * recall_rate / (precision_rate + recall_rate)
    print('precision_rate =', precision_rate, 'recall_rate =', recall_rate,
          'F1_measure =', f1_measure,
          file=output_file)
    print('precision_rate =', precision_rate, 'recall_rate =', recall_rate,
          'F1_measure =', f1_measure)
    return f1_measure


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
