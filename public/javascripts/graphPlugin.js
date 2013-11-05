function draw_graph(fileName) {

  sigInst = sigma.init($('#analysis')[0]).drawingProperties({
    defaultLabelColor: '#fff',
    defaultEdgeColor: '#70BCD6'
  }).graphProperties({
    minNodeSize: 2,
    maxNodeSize: 4
  }).mouseProperties({
    minRatio: 1,
    maxRatio: 3
  });

  sigInst.parseGexf(fileName);

   /**
   * Now, here is the code that shows the popup :
   */
  (function(){
    var popUp;

    // This function is used to generate the attributes list from the node attributes.
    // Since the graph comes from GEXF, the attibutes look like:
    // [
    //   { attr: 'Lorem', val: '42' },
    //   { attr: 'Ipsum', val: 'dolores' },
    //   ...
    //   { attr: 'Sit',   val: 'amet' }
    // ]
    function attributesToString(attr) {
      return '' +
        attr.map(function(o){
          return '' + o.attr + ' : ' + o.val + '';
        }).join('') +
        '';
    }

    function showNodeInfo(event) {
      popUp && popUp.remove();

      var node;
      sigInst.iterNodes(function(n){
        node = n;
      },[event.content[0]]);

      popUp = $(
        ''
      ).append(
        // The GEXF parser stores all the attributes in an array named
        // 'attributes'. And since sigma.js does not recognize the key
        // 'attributes' (unlike the keys 'label', 'color', 'size' etc),
        // it stores it in the node 'attr' object :
        attributesToString( node['attr']['attributes'] )
      ).attr(
        'id',
        'node-info'+sigInst.getID()
      ).css({
        'display': 'inline-block',
        'border-radius': 3,
        'padding': 5,
        'background': '#fff',
        'color': '#000',
        'box-shadow': '0 0 4px #666',
        'position': 'absolute',
        'left': node.displayX,
        'top': node.displayY+15
      });

      $('ul',popUp).css('margin','0 0 0 20px');

      $('#sigma-example').append(popUp);
    }

    function hideNodeInfo(event) {
      popUp && popUp.remove();
      popUp = false;
    }

    sigInst.bind('overnodes',showNodeInfo).bind('outnodes',hideNodeInfo).draw();
  })();

}