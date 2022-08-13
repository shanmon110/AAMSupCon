import re
with open('score.txt') as f:
    items = f.readlines()
    for item in items:
        num = re.findall(r"EER(.+?)bestEER",item)
        print(str(num[0]).replace(',',''))
