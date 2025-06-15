from flask import Flask, render_template, Response, jsonify
from detection import generate_frames, get_recent_logs, get_status

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_logs')
def get_logs():
    return jsonify(get_recent_logs())

@app.route('/get_status')
def status():
    return get_status()

if __name__ == '__main__':
    app.run(debug=True)
