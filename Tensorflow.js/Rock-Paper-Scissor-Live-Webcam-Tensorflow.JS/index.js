
let mobilenet;
let model;
// creates webcam object and point it to DOM object id='wc' 
const webcam = new Webcam(document.getElementById('wc'));
const dataset = new RPSDataset();
var rockSamples=0, paperSamples=0, scissorsSamples=0;
let isPredicting = false;


// get model into a object
async function loadMobilenet() {
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
  // get one of the output layers from pretrained mobilenet model
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  // we will create new model from  mobilenet and 'conv_pw_13_relu' as output
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

// new DNN that is used to classify with transfer learning

async function train() {
  // One-hot endoce the labels from the dataset before training
 
  dataset.ys = null;
  dataset.encodeLabels(3);
   
  model = tf.sequential({
    layers: [
      
      tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
      tf.layers.dense({ units: 100, activation: 'relu'}),
      tf.layers.dense({ units: 3, activation: 'softmax'})
    ]
  });
  const optimizer = tf.train.adam(0.0001);
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
  let loss = 0;
  model.fit(dataset.xs, dataset.ys, {
    epochs: 10,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        loss = logs.loss.toFixed(5);
        console.log('LOSS: ' + loss);
        }
      }
   });
}


// each button has an "id" [0,1,2] 

function handleButton(elem){
	switch(elem.id){
		case "0":
			rockSamples++;
			document.getElementById("rocksamples").innerText = "Rock samples:" + rockSamples;
			break;
		case "1":
			paperSamples++;
			document.getElementById("papersamples").innerText = "Paper samples:" + paperSamples;
			break;
		case "2":
			scissorsSamples++;
			document.getElementById("scissorssamples").innerText = "Scissors samples:" + scissorsSamples;
			break;
	}
    
	label = parseInt(elem.id);
    // capture contenct from a webcam
	const img = webcam.capture();
    
	dataset.addExample(mobilenet.predict(img), label);

}



async function predict() {
 
// 1. Predict class
  while (isPredicting) {
    
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);
      return predictions.as1D().argMax();
    });
      
// 2. Update user interface  
    const classId = (await predictedClass.data())[0];
    var predictionText = "";
    switch(classId){
		case 0:
			predictionText = "I see Rock";
			break;
		case 1:
			predictionText = "I see Paper";
			break;
		case 2:
			predictionText = "I see Scissors";
			break;
	}
	document.getElementById("prediction").innerText = predictionText;
			
// 3. clean up    
    predictedClass.dispose();
    await tf.nextFrame();
  }
}


function doTraining(){
	train();
}

// continous predictions - via html button and function call
function startPredicting(){
	isPredicting = true;
	predict();
}

// stop continous predictions - via html button and function call
function stopPredicting(){
	isPredicting = false;
	predict();
}

// Main function
async function init(){
	await webcam.setup();
    // load asynchronously mobilenet
	mobilenet = await loadMobilenet();
    // webcam.capture()  - grabs image from webcam in browser and converts it to tensor
    // mobilenet.predict() returns inferential
    // tf.tidy() - after the function => has finished it cleans all unused tensors exept the return one
	tf.tidy(() => mobilenet.predict(webcam.capture()));
		
}



init();
