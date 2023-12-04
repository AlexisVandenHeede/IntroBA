import json


with open('reportv1.ipynb', encoding="utf8") as json_file:
    data = json.load(json_file)

# print(data)
content = []
wordCount = 0
for each in data['cells']:
    cellType = each['cell_type']
    if cellType == "markdown":
        oldcontent = content.copy()
        content = each['source']
        if len(content) == 0:
            print(oldcontent)
            print('empty markdown cell after that one')
        if not content[0].strip().startswith("Collaboration"):
            for line in content:
                if line.startswith("# Appendix"):
                    break
                if line.startswith("### stuff to add after reading peer reviews"):
                    break
                if line.startswith("### Shape"):
                    break
                else:
                    temp = [word for word in line.split() if "#" not in word]
                    wordCount += len(temp)
# print(wordCount)

if wordCount > 2500:
    print(f'YOURE OVER GO DOWN {wordCount-2500} You are at {wordCount} words!')
else:
    print(f'Under 2500 words! You are at {wordCount} words!')
