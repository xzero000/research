import csv

data = []
file_num = 4
with open('g_all_test1_epoch_300000.csv','r' ) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data.append(row)

L = len(data)
l = int(L/file_num)
f = []
for i in range(file_num):
    if i == (file_num -1):
        ft = data[(l*i):]
    else:
        ft = data[l*i:l*(i+1)]
    f.append(ft)

for i in range(file_num):
    with open('g_all_test1_epoch_300000_%d.csv' %i,'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(f[i])


