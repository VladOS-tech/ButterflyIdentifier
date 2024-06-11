window.addEventListener('load', LaunchScripts)

const width = 720;
const height = 720;

function LaunchScripts(){
    let stream = null;
    StartStream();
    const photoButton = document.getElementById('photo-button-id');
    const canvas = document.getElementById('canvas-pillow');
    const inputFileButton = document.getElementById('file-button-id');
    const inputFileElement = document.getElementById('file-input-id');
    photoButton.addEventListener('click', () => takePic(canvas, stream));
    inputFileButton.addEventListener('click', () => inputFileElement.click());
    inputFileElement.addEventListener('change', () => {
        console.log(inputFileElement.files[0])
    })
}

function takePic(canvas, stream){
    canvas.width = width;
    canvas.height = height;
    let context = canvas.getContext('2d');
    context.drawImage(document.getElementById('camera-roll-id'), 0, 0, width, height)
    let dataURL = canvas.toDataURL("image/png")
    var newTab = window.open('about:blank','image from canvas');
    newTab.document.write("<img src='" + dataURL + "' alt='from canvas'/>");
}

async function StartStream(){
    const videoBlock = document.getElementById('camera-roll-id');
    try{
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: {ideal: width},
                height: {ideal: height},
                facingMode: {ideal: 'environment'}
            }
        });
        videoBlock.srcObject = stream;
    }
    catch(err){
        let img = document.createElement('img');
        img.src = "../assets/images/no cam.webp";
        img.alt = 'no camera';
        img.id = 'camera-roll-id'
        videoBlock.parentNode.replaceChild(img, videoBlock)
        alert(err);
    }
}