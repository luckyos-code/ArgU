import csv
from pymongo import MongoClient
csvfile = open('/home/ossama/Schreibtisch/args/query_sentiments.csv', 'r')
reader = csv.DictReader( csvfile )
mongo_client=MongoClient()
db=mongo_client.Args
db.segment.drop()
header= ["qid","text","sentiment_score","sentiment_magnitude" ]
for each in reader:
    row={}
    for field in header:
        row[field]=each[field]

    db.segment.insert(row)