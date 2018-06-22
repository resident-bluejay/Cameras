import csv;
#open file csv file

#r = csv.reader(open('camera_dataset.csv'), delimiter = ',')
r = csv.reader(open('dataset_test.csv'), delimiter = ',')
columns = [c for c in r]

#evaulate price and modify accordingly
for column in columns:
    a = column[12]
    if float(a) >= 1000:
        column[12] = '>1000'

    else:

        column[12] = '<=1000'

#write data into new file
writer = csv.writer(open('edited_dataset_test.csv', 'w'), delimiter = ',')
writer.writerows(columns)
