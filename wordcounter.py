import json


with open('report.ipynb') as json_file:
    data = json.load(json_file)

# print(data)

wordCount = 0
for each in data['cells']:
    cellType = each['cell_type']
    if cellType == "markdown":
        content = each['source']
        for line in content:
            temp = [word for word in line.split() if "#" not in word] # we might need to filter for more markdown keywords here
            wordCount = wordCount + len(temp)

# print(wordCount)

if wordCount > 2500:
    print(f'Matthias stop writing so much! You are at {wordCount} words!')
else:
    print(f'Under 2500 words! You are at {wordCount} words!')
