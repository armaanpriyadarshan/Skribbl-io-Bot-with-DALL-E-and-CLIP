import json

with open("words.json") as file:
    data = json.load(file)

class_file = open("C:\\Users\\armaa\\PycharmProjects\\Skribbl.io\\_tokenization.txt", "w")

for word in data:
    class_file.write(word + "\n")
