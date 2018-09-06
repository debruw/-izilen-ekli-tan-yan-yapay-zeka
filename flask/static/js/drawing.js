
// Initialize Variables
var path, ink;
var timer = 0, lastTimestamp = 0, lastTimestamp_check = 0;

// Install Paper.js
paper.install(window);

// Initialize...
window.onload = function() {

  initInk();              // Initialize Ink array ()
  paper.setup('canvas');  // Setup Paper #canvas

  var tool = new Tool();  // Inititalize Paper Tool

  // Paper Tool Mouse Down Event
  tool.onMouseDown = function(event) {
    // New Paper Path and Settings
    path = new Path();          
    path.strokeColor = 'black'; 
    path.strokeWidth = 5;

    // Get Time [ms] for each Guess
    var thisTimestamp = event.event.timeStamp;
    if(timer === 0){
      timer = 1; 
      var time = 0;
    }else{
      var timeDelta = thisTimestamp - lastTimestamp;
      var time = ink[2][ink[2].length-1] + timeDelta;
    }
    
    // Get XY point from event w/ time [ms] to update Ink Array
    updateInk(event.point, time);
    // Draw XY point to Paper Path
    path.add(event.point);
    
    // Reset Timestamps
    lastTimestamp = thisTimestamp;
  }

  // Paper Tool Mouse Drag Event
  tool.onMouseDrag = function(event) {
    // Get Event Timestamp and Timestamp Delta
    var thisTimestamp = event.event.timeStamp ;
    var timeDelta = thisTimestamp - lastTimestamp;
    // Get new Time for Ink Array
    var time = ink[2][ink[2].length-1] + timeDelta;
    
    // Get XY point from event w/ time [ms] to update Ink Array
    updateInk(event.point, time);
    // Draw XY point to Paper Path
    path.add(event.point);
    
    // Reset Timestamps
    lastTimestamp = thisTimestamp;

    // Check AI every 250 m/s
    if(thisTimestamp - lastTimestamp_check > 250){
      checkQuickDraw();
      lastTimestamp_check = thisTimestamp;
    }
  }

  // Initialize Info Modal
  initInfoModal();

}

// Initialize Ink Array
function initInk(){
  ink = [[],[],[]];
}

// Clear Paper Drawing Canvas
function clearDrawing() {
  // Remove Paper Path Layer
  paper.project.activeLayer.removeChildren();
  paper.view.draw();

  // Init Ink Array
  initInk();
  document.getElementById("myText").innerHTML = "";
  // Reset Variables
  timer = 0;
}

// Update Ink Array w/ XY Point + Time
function updateInk(point, time){
  ink[0].push(point.x);
  ink[1].push(point.y);
  ink[2].push(time);
}

// Get Paper Canvas Dimensions Width/Height
function getCanvasDimensions(){
  var w = document.getElementById('canvas').offsetWidth;
  var h = document.getElementById('canvas').offsetHeight;
  return {height: h, width: w};
}

// Check AI
function checkQuickDraw(){

  // Get Paper Canvas Weight/Height
  var c_dims = getCanvasDimensions();

  // Set Base URL for AI
  var url = 'http://localhost:5000/index'
  
  // Set HTTP Headers
  var headers = {
    'Accept': '*/*',
    'Content-Type': 'application/json'
  };

  // Init HTTP Request
  var xhr = new XMLHttpRequest();
  xhr.open('POST', url);
  Object.keys(headers).forEach(function(key,index) {
      xhr.setRequestHeader(key, headers[key]); 
  });

  // HTTP Request On Load
  xhr.onload = function() {
    if (xhr.status === 200) {
      res = xhr.responseText; // HTTP Response Text
      parseResponse(res);     // Parse Response
    }
    else if (xhr.status !== 200) {
      console.log('Request failed.  Returned status of ' + xhr.status);
    }
  };

  // Create New Data Payload for AI
  var data = {
    "language":"quickdraw",
    "writing_guide":{"width": c_dims.width, "height":c_dims.height},
    "ink":[ink]
    
  };

  // Convert Data Payload to JSON String
  var request_data = JSON.stringify(data);

  // Send HTTP Request w/ Data Payload
  xhr.send(request_data);
}


// Parse AI Response
function parseResponse(res){
  // Convert Response String to JSON
  var res_j = JSON.parse(res);
  document.getElementById("myText").innerHTML = res_j;
}

// Create and Fill Array
function createArray(len, itm) {
    var arr1 = [itm],
        arr2 = [];
    while (len > 0) {
        if (len & 1) arr2 = arr2.concat(arr1);
        arr1 = arr1.concat(arr1);
        len >>>= 1;
    }
    return arr2;
}

// Initialize Info Modal
function initInfoModal(){

  // Get the modal
  var modal = document.getElementById('info');

  // Get the button that opens the modal
  var btn = document.getElementById("btnInfo");

  // Get the <span> element that closes the modal
  var span = document.getElementsByClassName("close")[0];

  // When the user clicks on the button, open the modal 
  btn.onclick = function() {
      modal.style.display = "block";
  }

  // When the user clicks on <span> (x), close the modal
  span.onclick = function() {
      modal.style.display = "none";
  }

  // When the user clicks anywhere outside of the modal, close it
  window.onclick = function(event) {
      if (event.target == modal) {
          modal.style.display = "none";
      }
  }

  document.getElementById('info').style.display = "block";
  
}