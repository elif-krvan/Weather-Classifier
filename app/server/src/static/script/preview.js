const predictBtn = document.getElementById("predict");
const chosenFile = document.getElementById("chosen-file");
const resultMsg = document.getElementById("result");
const imagePreview = document.getElementById("image-preview");
const dropArea = document.getElementById("drop-area");
const fileSelector = document.getElementById("file-selector");

const darkBlueColor = "rgb(28, 79, 160)";
const dropAreaDefaultColor = "rgba(28, 79, 160, 0.272)";
const dropAreaOverColor = "rgba(28, 79, 160, 0.4)";

predictBtn.style.display = "none";

function previewImage(inputId = "file-selector") {
    predictBtn.style.display = "inline-block";
    const fileInput = document.getElementById(inputId);

    let uploadFile = fileInput.files;

    // remove border after image is uploaded
    imagePreview.style.border = "none";

    // clear previous view
    imagePreview.innerHTML = "";
    resultMsg.innerHTML = "";

    // Check if a file is selected
    if (uploadFile && uploadFile[0]) {
        const reader = new FileReader();

        reader.onload = function (e) {
            // create img element
            const img = document.createElement("img");
            img.src = e.target.result;
            imagePreview.appendChild(img);
        };

        // read selected file
        reader.readAsDataURL(uploadFile[0]);
        chosenFile.innerHTML = uploadFile[0].name;
    }
}

// drop area event listeners
imagePreview.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropArea.style.border = "3px dashed rgb(236, 236, 236)";
    dropArea.style.backgroundColor = dropAreaOverColor;
});

imagePreview.addEventListener("dragleave", () => {
    dropArea.style.border = `2px dashed ${darkBlueColor}`;
    dropArea.style.backgroundColor = dropAreaDefaultColor;
});

imagePreview.addEventListener("drop", (e) => {
    e.preventDefault();
    // imagePreview.style.border = "2px dashed #ccc";
    const files = e.dataTransfer.files;
    handleFiles(files);
});

function handleFiles(file) {
    // fire change event on the button file selector
    fileSelector.files = file;
    fileSelector.dispatchEvent(new Event("change"));
}
