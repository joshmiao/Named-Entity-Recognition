data_source = open("./data_source/data_source.txt", "r")

'''
the 'data' array contains 'word' in the following format:
    word : [text, type, organization_mark, place_mark, name_mark]
    text : {string}
    type : {string}
    organization_mark : {0, 1, 2} 
    place_mark : {0, 1, 2}
    name_mark : {0, 1, 2}
    0 for B(Begin), 1 for I(Inside), 2 for O(Outside)
'''
data = []

for lines in data_source:
    line_data = lines.split('  ')
    for word in line_data:
        data.append(word.split('/'))

for word in data:
    mark = [0, 0, 0]


