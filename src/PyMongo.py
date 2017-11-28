from pymongo import MongoClient
import json

with open("database/config.json","r") as f:
    config = json.load(f)

client = MongoClient('mongodb://%s:%s@' % (config["username"], config["password"])+config["url"]+'')
db = client[config["db_name"]]
collection = db['reviews']
document = collection.find_one({'_id': "sL2R9b1OnHkJiz25D3GTsg"})
print(document)
documents = collection.find({"business_id": "SMPbvZLSMMb7KU76YNYMGg"})
for document in documents:
    print(document)

client.close()