function displayGraph(graph) {
  var width = 960,
      height = 500;

  var color = d3.scale.category20();

  var force = d3.layout.force()
      .charge(-120)
      .linkDistance(30)
      .size([width, height]);

  var svg = d3.select("svg")
      .attr("width", width)
      .attr("height", height);


    force
        .nodes(graph.nodes)
        .links(graph.links)
        .start();

    var link = svg.selectAll(".link")
        .data(graph.links)
      .enter().append("line")
        .attr("class", "link")
        .style("stroke-width", function(d) { return Math.sqrt(d.value); });

    var node = svg.selectAll(".node")
        .data(graph.nodes)
      .enter().append("circle")
        .attr("class", "node")
        .attr("r", 5)
        .style("fill", function(d) { return color(d.group); })
        .call(force.drag);

    node.append("title")
        .text(function(d) { return d.name; });

    force.on("tick", function() {
      link.attr("x1", function(d) { return d.source.x; })
          .attr("y1", function(d) { return d.source.y; })
          .attr("x2", function(d) { return d.target.x; })
          .attr("y2", function(d) { return d.target.y; });

      node.attr("cx", function(d) { return d.x; })
          .attr("cy", function(d) { return d.y; });
    });
}


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
      displayGraph(message);
      //$('body').append('<p>'+JSON.stringify(message)+'</p>');
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
