from flask import Flask, request, redirect, render_template, url_for
from flask_sockets import Sockets
from werkzeug import secure_filename

import os
import trn
app = Flask(__name__, static_url_path="")
sockets = Sockets(app)

UPLOAD_FOLDER = "data/"
ALLOWED_EXTENSIONS = set(["txt", "csv", "dat"])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@sockets.route("/trn")
def trn_socket(ws):
    while True:
        print "Created WebSocket"
        message = ws.receive()
        print "Running TRN"
        ws.send('{"status":"Running TRN..."}')
        trnInstance = trn.main(UPLOAD_FOLDER + "wine_noannotation.csv", "35")
        jsonFile = open("graph.json")
        ws.send(jsonFile.read())

@app.route("/upload", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print "Posted"
        file = request.files['file']
        #codebook = request.args['codebook']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            trnInstance = trn.main(UPLOAD_FOLDER + filename, "35")
            jsonFile = open("graph.json")
            return jsonFile.read()
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="/upload" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

@app.route("/")
def hello():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
