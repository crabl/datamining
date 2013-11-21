function displayGraph(graph) {
  var width = $(window).width(),
      height = $(window).height();

  var sc_x = d3.scale.linear().domain([-4,4]).range([0, width]),
      sc_y = d3.scale.linear().domain([-4,4]).range([0, height]);

  var color = d3.scale.category20();

  var svg = d3.select("svg")
      .attr("width", width)
      .attr("height", height);

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
      .attr("cx", function(d) { return sc_x(d.x); })
      .attr("cy", function(d) { return sc_y(d.y); })
      .style("fill", function(d) { return color(d.group); });

  node.append("title")
      .text(function(d) { return d.name; });

  link.attr("x1", function(d) { return sc_x(graph.nodes[d.source].x); })
      .attr("y1", function(d) { return sc_y(graph.nodes[d.source].y); })
      .attr("x2", function(d) { return sc_x(graph.nodes[d.target].x); })
      .attr("y2", function(d) { return sc_y(graph.nodes[d.target].y); });
}


function constructTRN(fileName) {
/*  NProgress.start();
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
    sock.send(fileName);
  } else {
    sock.onopen = function(e) {
      sock.send(fileName);
    }
  }
*/
 
 var message = {"directed": false, "graph": [], "nodes": [{"y": -0.85408195088872962, "x": 1.8350508491713089, "id": 0}, {"y": 0.14869236018457224, "x": 2.9789479680876361, "id": 1}, {"y": -1.4691702046188555, "x": 0.036485824663521349, "id": 2}, {"y": 0.14976694677162061, "x": 2.9711234449967034, "id": 3}, {"y": 0.13349612800439062, "x": -3.1784097521652588, "id": 4}, {"y": 0.13847756164880959, "x": 2.9776898408580923, "id": 5}, {"y": 0.14037496980757697, "x": 2.9728369590791406, "id": 6}, {"y": -1.1878177969164703, "x": -1.3550847138101427, "id": 7}, {"y": 0.54106880852441619, "x": -3.5605338919295657, "id": 8}, {"y": 0.17192566887245672, "x": 2.9804997853822224, "id": 9}, {"y": 0.53703725129527913, "x": -3.5611066431935283, "id": 10}, {"y": -1.2026832349978716, "x": -1.3276823552117611, "id": 11}, {"y": 0.51540887752664954, "x": -3.5365034556066623, "id": 12}, {"y": 0.27479474054734948, "x": 2.9826334919621931, "id": 13}, {"y": -0.57574546431671125, "x": -2.3778528218915818, "id": 14}, {"y": 0.13448945047957483, "x": 2.9746478930521425, "id": 15}, {"y": 0.21905735228307735, "x": 2.981880202773191, "id": 16}, {"y": 0.54861004850386053, "x": -3.5597599854785047, "id": 17}, {"y": 0.15641025441319342, "x": 2.979599691973525, "id": 18}, {"y": -0.36938309901862432, "x": 2.4621048879664476, "id": 19}, {"y": 0.53287373530530369, "x": -3.5622058071166625, "id": 20}, {"y": -1.1981077846221673, "x": -1.3261719611603524, "id": 21}, {"y": -1.0096127149322842, "x": -1.717727147040226, "id": 22}], "links": [{"source": 0, "target": 2, "weight": 1}, {"source": 0, "target": 19, "weight": 1}, {"source": 1, "target": 18, "weight": 1}, {"source": 1, "target": 5, "weight": 1}, {"source": 2, "target": 21, "weight": 1}, {"source": 3, "target": 6, "weight": 1}, {"source": 4, "target": 12, "weight": 1}, {"source": 4, "target": 14, "weight": 1}, {"source": 5, "target": 19, "weight": 1}, {"source": 5, "target": 15, "weight": 1}, {"source": 6, "target": 15, "weight": 1}, {"source": 7, "target": 11, "weight": 1}, {"source": 7, "target": 21, "weight": 1}, {"source": 7, "target": 22, "weight": 1}, {"source": 8, "target": 17, "weight": 1}, {"source": 8, "target": 10, "weight": 1}, {"source": 8, "target": 12, "weight": 1}, {"source": 9, "target": 16, "weight": 1}, {"source": 9, "target": 18, "weight": 1}, {"source": 10, "target": 20, "weight": 1}, {"source": 12, "target": 17, "weight": 1}, {"source": 13, "target": 16, "weight": 1}, {"source": 14, "target": 22, "weight": 1}], "multigraph": false};


  displayGraph(message);
}
