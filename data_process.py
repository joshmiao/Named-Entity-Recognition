import torch


def preprocess_data(data_dir, output_file=None):
    """
    the returned 'data' list contains 'word's in the following format:
        word : [text, type, organization_mark, place_mark, name_mark]
            text : {string}
            type : {string}
            organization_mark : {0, 1, 2}
            place_mark : {0, 1, 2}
            name_mark : {0, 1, 2}
                 0 for O(Outside), 1 for B(Begin), 2 for I(Inside)
    """
    data = list()
    data_source = open(data_dir, 'r')

    # processing and marking data
    for lines in data_source:
        line_data = lines.split('  ')
        for word in line_data:
            if len(word.split('/')) == 2:
                if word.split('/')[0].find('199801') == -1:
                    data.append(word.split('/'))
                else:
                    data.append(['#开头#', 'm'])

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
            data[idx][0] = data[idx][0].split('[')[1]
            # the data source can be error
            if end_id - idx > 10:
                continue
            # process name and type
            ty = data[end_id][1].split(']')[1]
            data[end_id][1] = data[end_id][1].split(']')[0]
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
    if output_file is not None:
        data_output = open(output_file, "w")
        print(data, file=data_output)
    return data


def make_dictionary(dict_size, data):
    """
    making dictionary of commonly used words
    return a dictionary mapping words to its id
    """
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
    return dic


def create_tensor_list(dic, data, device):
    """
    creating tensor list
    x_tlist is a one-hot encoding list of tensors
    y_tlist is the corresponding three labels in {0, 1, 2}
    """
    x_tlist = list()
    y_tlist = list()
    dict_size = len(dic)
    print('Total', len(data), 'words')
    for idx in range(1, len(data) - 1):
        if data[idx][0] == '#开头#' or data[idx + 1][0] == '#开头#':
            continue
        x = [0] * (dict_size + 1) * 3
        label = [data[idx][2], data[idx][3], data[idx][4]]
        x[dic.get(data[idx - 1][0], 0)] = 1
        x[dic.get(data[idx][0], 0) + (dict_size + 1)] = 1
        x[dic.get(data[idx + 1][0], 0) + 2 * (dict_size + 1)] = 1
        x_tlist.append(torch.tensor(x, device=device, dtype=torch.float32))
        y_tlist.append(torch.tensor(label, device=device, dtype=torch.int))
    return x_tlist, y_tlist
