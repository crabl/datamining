function getTRNData() {
  NProgress.start();
  var sock = new WebSocket("ws://"+document.domain+"/trn");

  // For local development
  if(document.location.port) {
    sock = new WebSocket("ws://"+document.domain+":"+document.location.port+"/trn");
  }

  sock.onmessage = function(msg) {
    var message = JSON.parse(msg.data);
    if(message.nodes) {
      console.log("TRN DATA DETECTED")
      NProgress.done();
      $('body').append('<p>'+JSON.stringify(message)+'</p>');
    } else {
      NProgress.set(message.progress/100);
    }

  };
  if(sock.readyState == 1) {
    sock.send('some data');
  } else {
    sock.onopen = function(e) {
        sock.send('some data');
    }
  }
}
