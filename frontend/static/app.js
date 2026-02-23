const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const loader = document.getElementById('loader');
const resultsSection = document.getElementById('resultsSection');
const predBadge = document.getElementById('predBadge');
const downloadBtn = document.getElementById('downloadBtn');
const customAlert = document.getElementById('customAlert');

// --- Helper: Show Alert ---
function showAlert(message, type = 'info') {
    customAlert.textContent = message;
    customAlert.style.display = 'block';
    customAlert.style.color = type === 'error' ? '#ff5252' : '#00c0a3';
    setTimeout(() => {
        customAlert.style.display = 'none';
    }, 5000);
}

// --- Drag & Drop Handlers ---
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, e => {
        e.preventDefault();
        e.stopPropagation();
    }, false);
});

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add('active'), false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove('active'), false);
});

dropZone.addEventListener('drop', e => {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
});

fileInput.addEventListener('change', e => {
    handleFiles(e.target.files);
});

async function handleFiles(files) {
    const file = files[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
        showAlert("Please upload a valid medical image file.", "error");
        return;
    }

    // Reset UI
    resultsSection.classList.remove('show');
    loader.style.display = 'block';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/analyze', { method: 'POST', body: formData });
        if (!response.ok) throw new Error("Diagnostic analysis failed. Please try again.");

        const result = await response.json();

        // Update Images
        document.getElementById('originalImg').src = `data:image/png;base64,${result.original_image}`;
        document.getElementById('gradcamImg').src = `data:image/png;base64,${result.gradcam_image}`;

        // Update Metrics
        predBadge.textContent = result.label;
        predBadge.className = `prediction-badge ${result.is_malignant ? 'malignant' : 'benign'}`;

        document.getElementById('confPercent').textContent = result.percent.toFixed(2) + "%";
        document.getElementById('classId').textContent = result.class_id;

        // Show Results with Animation
        loader.style.display = 'none';
        resultsSection.classList.add('show');
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch (err) {
        loader.style.display = 'none';
        showAlert(err.message, "error");
    }
}

// --- Report Generation ---
downloadBtn.addEventListener('click', async () => {
    const file = fileInput.files[0] || null;
    if (!file) {
        showAlert("No file available for report generation.", "error");
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const originalText = downloadBtn.innerHTML;
        downloadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating Report...';
        downloadBtn.disabled = true;

        const response = await fetch('/report', { method: 'POST', body: formData });
        if (!response.ok) throw new Error("Report generation failed.");

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `Thyroid_Report_${new Date().getTime()}.docx`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);

        downloadBtn.innerHTML = originalText;
        downloadBtn.disabled = false;
        showAlert("Clinical report downloaded successfully.");
    } catch (err) {
        showAlert(err.message, "error");
        downloadBtn.disabled = false;
    }
});
