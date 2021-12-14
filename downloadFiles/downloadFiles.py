from downloadUtils import *
from pymongo import MongoClient
from datetime import datetime as dt
from datetime import timedelta
import pandas as pd
import numpy as np
import os
from bson import ObjectId
from google.cloud import storage
from multiprocessing import Pool


def getFile(session, component, bearing_num, sample):
    plane = sample['plane']
    sensor = sample['sensorType'].split('_')[0].lower()

    dest_filename = '{}_{}_bearing_{}_plane_{}_{}.npy'.format(
        str(session['_id']), component['type'], bearing_num, plane, sensor)
    path = os.path.join(dest_dir, dest_filename)
    if os.path.isfile(path):
        print(path + " already exist, skipping")
        return

    dataurl = sample['dataUrl'].lstrip('/')
    try:
        sample_data = bucket.blob(dataurl).download_as_string()
        data = bytes_unzip(sample_data, temp_file="/tmp/temp.zip")
        header, data = parse_xml(data)
        twf = load_raw_waveform(data, header['dataFormat'], header['Frequency'])
        twf_calab = calibrat_data(twf[1], header)
        arr = np.asarray([twf[0], twf_calab])
        np.save(path, arr)
        print(
            "Successfully downloaded session {} {} bearing {} plane {} {} sample ({})".format(
                str(session['_id']), component['type'], bearing_num, plane, sensor, dataurl)
        )
    except:
        print(
            "Failed downloading session {} {} bearing {} plane {} {} sample ({})".format(
                str(session['_id']), component['type'], bearing_num, plane, sensor, dataurl)
        )
    return

if __name__ == '__main__':

    # mongo_client = MongoClient(host=os.environ['mongo_string'])
    mongo_client = MongoClient(
        "mongodb://algo-research:mTLJKBZcftGbVuEk@production-gcp-shard-00-00-kxuwc.gcp.mongodb.net:27017,"
        "production-gcp-shard-00-01-kxuwc.gcp.mongodb.net:27017,"
        "production-gcp-shard-00-02-kxuwc.gcp.mongodb.net:27017/production"
        "?ssl=true&replicaSet=production-gcp-shard-0&authSource=admin&readPreference=secondary"
    )
    mongo_db = mongo_client['production']

    gcs_client = storage.Client(project="research-150008")
    bucket = gcs_client.bucket("fileserver-service-production")

    dest_dir = "/Users/gkogan/Downloads/newData"

    dt_obj_start = dt.strptime('2021-10-01 0:0:0', '%Y-%m-%d %H:%M:%S')
    dt_obj_end = dt_obj_start + timedelta(hours=6 * 24)
    machine_id = '6022c7d2dbf9880001a364e9'

    data = list(
        mongo_db.sessions.find(
            {'machineId': ObjectId(machine_id),
             'created_at': {
                 '$gt': dt(
                     dt_obj_start.year,
                     dt_obj_start.month,
                     dt_obj_start.day,
                     dt_obj_start.hour,
                     dt_obj_start.minute,
                     dt_obj_start.second
                 ),
                 '$lt': dt(
                     dt_obj_end.year,
                     dt_obj_end.month,
                     dt_obj_end.day,
                     dt_obj_end.hour,
                     dt_obj_end.minute,
                     dt_obj_end.second
                 )
             }}))

    # iterate the data and append to a list the machine id and the session id
    sessionsData = [[str(d['machineId']), str(d['_id']), d['created_at']] for d in data]
    dfSessionsData = pd.DataFrame(sessionsData, columns=['machine_id', 'session_id', 'timestamp'])

    session_data = mongo_db.sessions.find({"_id": {"$in": [ObjectId(s) for s in dfSessionsData.session_id.values]}})
    # [
    #     getFile(session, component, bearing_num, sample)
    #     for session in session_data
    #     for component in session['components']
    #     for bearing_num, bearing in enumerate(component.get('bearings', []))
    #     for sample in bearing.get('samples', [])
    # ]

    with Pool(processes=100) as pool:
        multiple_results = [
            pool.apply_async(getFile, args=(session, component, bearing_num, sample))
            for session in session_data
            for component in session['components']
            for bearing_num, bearing in enumerate(component.get('bearings', []))
            for sample in bearing.get('samples', [])
        ]
        [res.get() for res in multiple_results]



