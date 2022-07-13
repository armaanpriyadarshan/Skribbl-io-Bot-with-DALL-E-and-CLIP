import json

with open("words.json") as file:
    data = json.load(file)

i = 1
for key in data:
    if i == 2115:
        print(key)
        break
    i += 1
