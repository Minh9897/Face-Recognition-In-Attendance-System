from flask import Flask, render_template, request, Response, jsonify
import cv2,time, pickle, json, os, re
from app.export_csv import markAttendance, create_table_csv
from app.video_process import Processing
from app.train import Train_Model_Thread as train_thread
import numpy as np
from datetime import datetime
from configparser import ConfigParser

config = ConfigParser()
config.read('app/config.ini')
with open('app/date.pkl', 'rb') as f:
    date_flag = pickle.load(f)
    f.close()

is_running = False    
threshold = np.loadtxt(config.get('path','threshold'), dtype=int)
threshold_value = {'face_detect_threshold':int(threshold[0]),
					'small_face_delete_threshold':int(threshold[1]),
					'face_verification_threshold':int(threshold[2])}

app = Flask(__name__)

def videoProcess_thread():
	global process_thread
	process_thread = Processing(threshold_value)
	process_thread.start()

def get_date_time_now():
	now = datetime.now()  
	return now.strftime("%d_%m_%Y")

def video_process():
	global is_running
	if is_running == False:
		videoProcess_thread()
		is_running = True
	while(True):
		global date_flag
		date_now = get_date_time_now()
		if date_now != date_flag:
			create_table_csv()
			with open('app/date.pkl', 'wb') as f: 
				pickle.dump(date_now, f)
				f.close()
			date_flag = date_now
		process_thread.get_threshold(threshold_value)
		ret, img = process_thread.get_frame(state)
		if ret == True:
			now = datetime.now()
			dtString = now.strftime('%H:%M:%S')
			if state == "verification":
				name_list = process_thread.get_name_list()
				if len(name_list) > 0:
					for name in name_list:
						markAttendance(name,dtString)			
			frame = cv2.imencode('.JPEG', img)[1].tobytes()
			time.sleep(0.016)
			yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
		else:
			pass

def load_table(date):
    date_time = date.strftime("%d_%m_%Y")
    fname = config.get('output','csv_path') + "Attendance_" + date_time +".csv"
    with open(fname,'r+') as f:   
        myDataList = f.readlines()
        table = []
        for line in myDataList:
            entry = line.split(',')
            entry[-1] = entry[-1].replace('\n','')
            table.append(tuple(entry))
    return table

def list_csv():
    arr = os.listdir('./app/data/')
    file =[]
    for item in arr:
        date = [int(s) for s in re.findall('\\d+',item)]
        date = datetime(date[2],date[1],date[0])
        file.append(date)
    file.sort(reverse=True)
    list_datetime = file[:5]
    list_date = [item.strftime("%d/%m/%Y") for item in list_datetime]
    return list_date,list_datetime


@app.route('/')
def index():
	global state
	state = 'verification'
	return render_template("results.html",threshold_value=threshold_value)

@app.route('/mode_change',methods = ['POST'])
def mode_change():	
	global state
	state = request.form['mode']
	return "changed"

@app.route('/res_threshold',methods = ['POST'])
def res_threshold():
	global threshold_value
	threshold_value = request.form.to_dict()
	return 'changed'

@app.route('/save_threshold',methods = ['POST'])
def save_threshold():
	threshold = np.array([int(item) for item in threshold_value.values()])
	np.savetxt(config.get('path','threshold'),threshold,'%i') 
	return 'save'

@app.route('/results')
def video_feed():
	return Response(video_process(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/training')
def train_model():
	global state
	state = None
	return render_template("training.html")

@app.route('/start_training', methods = ['POST'])
def start_training():
	global trainThread
	trainThread = train_thread()
	trainThread.start()
	return "training"

@app.route("/listen")
def listen():
	def respond_to_client():
		while True:
			try:
				text = trainThread.get_process()
				_data = json.dumps({'process': text})
			except IOError:
				_data = json.dumps({'process': ''})                
			yield f"id: 1\ndata: {_data}\nevent: online\n\n"
			time.sleep(0.5)
			if text == "Done!":
				break
	return Response(respond_to_client(), mimetype='text/event-stream')

@app.route('/record_status', methods=['POST'])
def record_status():
    global process_thread
    if process_thread == None:
        process_thread = Processing(threshold_value)

    json = request.get_json()

    status = json['status']

    if status == "true":
        train_name = json['train_name']
        process_thread.start_record(train_name)
        return jsonify(result="started")
    else:
        process_thread.stop_record()
        return jsonify(result="stopped")

@app.route("/index", methods=["POST","GET"])
def table():
    global date
    fields = ['Name', 'Time']
    list_date, list_datetime = list_csv()
    if request.method == "POST":
        params = request.form.to_dict()
        date3 = datetime.strptime(params['date3'], "%Y-%m-%d")
        date2 = datetime.strptime(params['date2'], "%d/%m/%Y")
        if date3 != date2:
            date = date3
            if date not in list_datetime:
                list_datetime.append(date)
                list_datetime.sort(reverse=True)
                list_date = [item.strftime("%d/%m/%Y") for item in list_datetime]
        else:
            date = date2
    else:
        date = datetime.now()
        date_now = date.strftime("%d/%m/%Y")
        if date_now not in list_date:
            list_date.insert(0,date_now)
    selected_date = date    
    return render_template("index.html",headings = fields, list_date = list_date,selected_date=selected_date)

@app.route("/listen_database")
def listen_database():
    def respond_to_client():
        while True:
            try:
                row = load_table(date)
                if len(row) > 1:
                    _data = json.dumps({ 'row': row[1:], 'nodata':'False'})
                else: _data = json.dumps({'nodata': 'True'})
            except IOError:
                _data = json.dumps({'nodata': 'True'})                 
            yield f"id: 1\ndata: {_data}\nevent: online\n\n"
            time.sleep(0.5)
    return Response(respond_to_client(), mimetype='text/event-stream')

if __name__ == "__main__":
	create_table_csv()
	now = get_date_time_now()
	app.run(debug=True, host='192.168.0.106',port=9999)
