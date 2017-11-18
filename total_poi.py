### Counting the total persons of interest in the poi_names.txt file.
### The y and n tags in front of the names are not referring to poi.
### All of the names in this file are poi so just count all names.

count = 0

with open('poi_names.txt', 'r') as f:
    for line in f:
        if '(y)' in line or '(n)' in line:
            count += 1

print count