document.getElementById("uploadForm").onsubmit = async function (e) {
  e.preventDefault();

  const fileInput = document.getElementById("fileInput");
  const analyzeButton = document.getElementById("analyzeButton");

  fileInput.disabled = true;
  analyzeButton.disabled = true;

  document.getElementById("hourglass").style.display = "inline-block";
  document.getElementById("progressContainer").style.display = "block";
  document.getElementById("progressContainer").style.width = "450px";

  // Upload file and start process
  const fd = new FormData();
  fd.append("file", fileInput.files[0]);

  await fetch("/start-process", { method: "POST", body: fd });

  // Poll progress
  const interval = setInterval(async () => {
    const res = await fetch("/progress");
    const prog = await res.json();

    document.getElementById("mainStatus").innerText = prog.stage;
    document.getElementById("main-bar").style.width = prog.percent + "%";

    if (prog.percent >= 100) {
      clearInterval(interval);
      window.location.href = "/results";
    }
  }, 300);
};

// Custom button triggers hidden input
document.getElementById("customFileButton").onclick = function () {
  document.getElementById("fileInput").click();
};

// Update label when a file is selected
document.getElementById("fileInput").onchange = function () {
  const label = document.getElementById("fileNameLabel");
  if (this.files.length > 0) {
    label.innerText = this.files[0].name;
  } else {
    label.innerText = "No file chosen";
  }
};
