function getTRNData() {
  var sock = new WebSocket("ws://"+document.domain+"/trn");
  sock.onmessage = function(msg) {
    var message = JSON.parse(msg.data);
    if(message.nodes) {
      console.log("TRN DATA DETECTED")
    }
    console.log(msg.data);

  };
  if(sock.readyState == 1) {
    sock.send('some data');
  } else {
    sock.onopen = function(e) {
        sock.send('some data');
    }
  }
}
