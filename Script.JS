// Setup canvas drawing context
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let isDrawing = false;

// Handle drawing on canvas
canvas.addEventListener("mousedown", (e) => {
    isDrawing = true;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
});

canvas.addEventListener("mousemove", (e) => {
    if (isDrawing) {
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
    }
});

canvas.addEventListener("mouseup", () => {
    isDrawing = false;
});

// Clear canvas
document.getElementById("clear").addEventListener("click", () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById("result").innerText = "";
});

// Predict digit when button is clicked
document.getElementById("predict").addEventListener("click", async () => {
    const imageData = canvas.toDataURL("image/png");

    // Preprocess image and run the prediction
    const digit = await predictDigit(imageData);

    document.getElementById("result").innerText = `Predicted digit: ${digit}`;
});

async function predictDigit(imageData) {
    // Preprocess the image data (convert to grayscale, resize to 28x28)
    const img = await loadImage(imageData);
    const tensor = preprocessImage(img);

    // Load the model (the model.json should be pre-trained and available)
    const model = await tf.loadLayersModel('model/model.json');  // Change this path as necessary

    // Run prediction
    const prediction = model.predict(tensor);
    const predictedClass = prediction.argMax(1).dataSync()[0];

    return predictedClass;
}

function loadImage(imageData) {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.src = imageData;
    });
}

function preprocessImage(img) {
    // Resize the image to 28x28 pixels
    const canvas2 = document.createElement('canvas');
    const ctx2 = canvas2.getContext('2d');
    canvas2.width = 28;
    canvas2.height = 28;
    ctx2.drawImage(img, 0, 0, 28, 28);

    // Convert image to grayscale (28x28x1)
    const imgData = ctx2.getImageData(0, 0, 28, 28);
    const data = imgData.data;
    const grayscaleData = new Float32Array(28 * 28);

    for (let i = 0; i < data.length; i += 4) {
        const grayscaleValue = 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
        grayscaleData[i / 4] = grayscaleValue / 255; // Normalize between 0 and 1
    }

    // Create a tensor from the grayscale data
    return tf.tensor(grayscaleData, [1, 28, 28, 1]);
}
