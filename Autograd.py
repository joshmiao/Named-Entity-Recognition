import torch
import numpy as np
data_source = open('./data_source/data_source.txt', 'r')

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

dict_size = 50
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


# device = torch.device('cpu')
device = torch.device('cuda:0')
x_tlist = list()
y_tlist = list()
empty_list = [0] * (dict_size + 1) * 3
print('Total', len(data), 'words')
for idx in range(1, len(data) - 1):
    x = empty_list.copy()
    label = [data[idx][2], data[idx][3], data[idx][4]]
    x[dic.get(data[idx - 1][0], 0)] = 1
    x[dic.get(data[idx][0], 0) + (dict_size + 1)] = 1
    x[dic.get(data[idx + 1][0], 0) + 2 * (dict_size + 1)] = 1
    x_tlist.append(torch.tensor(x, device=device, dtype=torch.float))
    y_tlist.append(torch.tensor(label, device=device, dtype=torch.int))

print('done')
#################################################################################

if len(x_tlist) != len(y_tlist):
    print('Data error!')

sample_cnt = len(x_tlist)
batch_size = sample_cnt
epoch = 100

theta = torch.zeros(3, 3 * (dict_size + 1), device=device, dtype=torch.float, requires_grad=True)

for __epoch__idx__ in range(epoch):
    lo = torch.zeros(1, device=device, dtype=torch.float)
    for idx in range(sample_cnt // 10):
        sigma = torch.zeros(1, device=device, dtype=torch.float)
        for i in range(3):
            sigma += torch.exp(theta[i] @ x_tlist[idx])
        sigma = torch.log(sigma)
        lo += theta[y_tlist[idx][0]] @ x_tlist[idx] - sigma
    lo.backward(retain_graph=True, gradient=torch.ones(1, dtype=torch.float32, device=device))
    print(theta.grad)
