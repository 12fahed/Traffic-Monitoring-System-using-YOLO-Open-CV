from flask import Flask, render_template, Response, url_for, jsonify  
import main  # Your vehicle detection script

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/traffic_signals')
def traffic_signals():
    return render_template('traffic_signals.html')

# Routes for each video feed
@app.route('/video_feed1')
def video_feed1():
    return Response(main.generate_frames(1), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(main.generate_frames(2), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed3')
def video_feed3():
    return Response(main.generate_frames(3), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed4')
def video_feed4():
    return Response(main.generate_frames(4), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed5')
def video_feed5():
    return Response(main.generate_frames(5), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed6')
def video_feed6():
    return Response(main.generate_frames(6), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed7')
def video_feed7():
    return Response(main.generate_frames(7), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed8')
def video_feed8():
    return Response(main.generate_frames(8), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/getVideoCount')
def get_video_count():
    return jsonify(main.vehicle_count)  # Return vehicle_count as JSON

if __name__ == "__main__":
    app.run(debug=True)
