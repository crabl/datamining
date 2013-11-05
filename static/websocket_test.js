function getTRNData() {
  var sock = new WebSocket("ws://"+document.domain+":5000/trn");
  sock.onmessage = function(msg) { console.log(msg.data); };
  if(sock.readyState == 1) {
    sock.send('some data');
  } else {
    sock.onopen = function(e) {
        sock.send('some data');
    }
  }
}
