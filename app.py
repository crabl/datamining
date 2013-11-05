from flask import Flask, request, redirect, render_template, url_for
from flask_sockets import Sockets
from werkzeug import secure_filename

import trn

app = Flask(__name__)
sockets = Sockets(app)

UPLOAD_FOLDER = "data/"
ALLOWED_EXTENSIONS = set(["txt", "csv", "dat"])

@sockets.route("/trn")
def trn_socket(ws):
    while True:
        message = ws.receive()
        trnInstance = trn.main(UPLOAD_FOLDER + "/wine_noannotation.csv", "35")
        jsonFile = open("graph.json")
        ws.send(jsonFile.read())
        ws.send("ANOTHER ECHO!" + message)


@app.route("/upload", methods=["POST"])
def upload_file():

    return "OK"

@app.route("/")
def hello():
    return render_template("index.html")

if __name__ == "__main__":
    app.run()
