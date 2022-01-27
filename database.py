import pandas as pd
import pymongo
from pymongo import MongoClient
#pipe agregate funtction #############################################
def get_all(data):
        print('\nCreating database connection...')
        client = MongoClient("localhost:27017")
        db = client["assignment2"]
        collection = db[data]
        print('\nGetting data...')
        return pd.DataFrame(list(collection.find({})))

def upload_data(df):
    print('\nCreating database connection...')
    conn = MongoClient("localhost:27017")
    db = conn.assignment2
    collection = db.hotel_reviews
    print('\nTrying to insert the data...')
    collection.insert_many(df.to_dict('records'))
    print('\nSuccessful uploaded data')

def aggregate_fun():
    print('\nCreating database connection...')
    client = MongoClient("localhost:27017")
    db = client["assignment2"]
    
    print('\nApplying aggregate function...')
    pipeline = [
    {
        "$match": { 'Total_Number_of_Reviews': { '$gt': 5000 } }
    },
    ]
    return pd.DataFrame(db.data_hotels.aggregate(pipeline))