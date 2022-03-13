data_source = open("./data_source/data_source.txt", "r")

'''
the 'data' list contains 'word' in the following format:
    word : [text, type, organization_mark, place_mark, name_mark]
    text : {string}
    type : {string}
    organization_mark : {0, 1, 2} 
    place_mark : {0, 1, 2}
    name_mark : {0, 1, 2}
    0 for O(Outside), 1 for B(Begin), 2 for I(Inside)
'''
data = []

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
    mark.append(1) if word[1] == 'nr' else mark.append(0)
    word += mark

idx = 0
while idx < len(data):
    if data[idx][0].find('[') != -1:
        end_id = idx + 1
        while data[end_id][1].find(']') == -1:
            end_id += 1
        ty = data[end_id][1].split(']')[1]
        data[end_id][1] = data[end_id][1].split(']')[0]
        data[idx][0] = data[idx][0].split('[')[1]
        if ty == 'nt':
            data[idx][2] = 1
        elif ty == 'ns':
            data[idx][3] = 1
        elif ty == 'nr':
            data[idx][4] = 1
        idx += 1
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
data_output = open("./data_source/data_output.txt", "w")
print(data, file=data_output)
