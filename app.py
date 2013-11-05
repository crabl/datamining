from flask import Flask
from flask_sockets import Sockets

app = Flask(__name__)
sockets = Sockets(app)

@sockets.route("/echo")
def echo_socket(ws):
    while True:
        message = ws.receive()
        ws.send("From the server, " + message)

@app.route("/")
def hello():
    return "Hello, world!"

if __name__ == "__main__":
    app.run()
