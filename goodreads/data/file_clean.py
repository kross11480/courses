
import re
pattern = re.compile("<(\d{4,5})>")

authors = []
for i, line in enumerate(open('reading_list.txt')):
    str = line.split(':')
    if len(str) > 1:
        author_surname = str[0].split(' ')[-1]
        authors.append(author_surname) 

print (authors)