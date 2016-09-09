import csv
import pandas as pd

# reader1 = csv.reader(open('myfirstforest.csv', 'rb'))
# reader2 = csv.reader(open('data/output.csv', 'rb'))
# reader3 = csv.reader(open('data/gptest.csv', 'rb'))
#
# result1 = dict()
# result2 = dict()
# result3 = dict()
#
# reader1.next()
# for row in reader1:
#     result1[row[0]] = row[1]
#
# reader2.next()
# for row in reader2:
#     result2[row[0]] = row[1]
#
# reader3.next()
# for row in reader3:
#     result3[row[0]] = row[1]
#
# cnt = 0
# for key in result1:
#     print key
#     if result1[key] == result2[key] and result1[key] == result3[key]:
#         cnt +=1
#         # print key
#
# # print len(result1)
# # print cnt

db1 = pd.read_csv('myfirstforest.csv', header=0)
db2 = pd.read_csv('data/output.csv', header=0)
db3 = pd.read_csv('data/gptest.csv', header=0)

cnt = 0
ids = dict()
new_data = []
for _id in range(db1.shape[0]):
    if db1.iloc[_id]['Survived'] == db2.iloc[_id]['Survived'] and db1.iloc[_id]['Survived'] == db3.iloc[_id]['Survived']:
        # print db1.iloc[_id]['PassengerId']
        ids[db1.iloc[_id]['PassengerId']] = db1.iloc[_id]['Survived']
        new_data.append(db1.iloc[_id]['Survived'])
    else:
        new_data.append(2)
new_df = pd.DataFrame({'Survived': new_data}, dtype=int)

train_db = pd.read_csv('data/train.csv', header=0)
test_db = pd.read_csv('data/test.csv', header=0)

test_db = pd.concat([test_db, new_df], axis=1)
# test_db['Survived'] = test_db['Survived'].map({0.0: 0, 1.0: 1, None: None}).astype(int)
# print test_db[test_db.Survived != 2]

header = []
for att in train_db:
    header.append(str(att))

print header

new_train = pd.concat([train_db, test_db[test_db.Survived != 2]], axis=0)

new_train[header].to_csv('data/new_train.csv', index=False)