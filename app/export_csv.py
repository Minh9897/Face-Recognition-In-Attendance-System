from datetime import datetime
import csv
from configparser import ConfigParser
config = ConfigParser()
config.read('app/config.ini')

#Create file csv
def create_table_csv():
    now = datetime.now()
    fields = ['Name', 'Time']
    date_time = now.strftime("%d_%m_%Y")
    fname = config.get('output','csv_path') + "Attendance_" + date_time +".csv"
    try:
        f = open(fname)
        # Do something with the file
    except IOError:
        with open(fname,'w',newline="") as f:
            write = csv.writer(f)
            write.writerow(fields)
    finally:
        f.close()

def markAttendance(name,dtString):
    now = datetime.now()
    date_time = now.strftime("%d_%m_%Y")
    fname = config.get('output','csv_path') + "Attendance_" + date_time +".csv"
    with open(fname,'r+') as f:   
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            f.writelines(f'{name},{dtString}\n')