count = 0
total = 0
with open('data/reviews.csv', 'r') as f:
    for s in f:
        total += 1
        tokens = s.split(',')
        if len(tokens) != 9:
            count += 1
print 'Total lines:', total
print 'Number of errors :',count