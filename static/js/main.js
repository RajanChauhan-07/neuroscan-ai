// ── Upload Zone ───────────────────────────────────────────
const uploadZone    = document.getElementById('uploadZone');
const mriInput      = document.getElementById('mriInput');
const uploadContent = document.getElementById('uploadContent');
const uploadPreview = document.getElementById('uploadPreview');
const previewImg    = document.getElementById('previewImg');
const previewName   = document.getElementById('previewName');

uploadZone.addEventListener('click', () => mriInput.click());

uploadZone.addEventListener('dragover', e => {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
});
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    const dt = new DataTransfer();
    dt.items.add(file);
    mriInput.files = dt.files;
    showPreview(file);
  }
});

mriInput.addEventListener('change', e => {
  if (e.target.files[0]) showPreview(e.target.files[0]);
});

function showPreview(file) {
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    previewName.textContent = file.name;
    uploadContent.style.display = 'none';
    uploadPreview.style.display = 'flex';
  };
  reader.readAsDataURL(file);
}

document.querySelector('.preview-change')?.addEventListener('click', e => {
  e.stopPropagation();
  mriInput.click();
});

// ── Form Submit ───────────────────────────────────────────
const form          = document.getElementById('analysisForm');
const loadingOverlay = document.getElementById('loadingOverlay');
const results       = document.getElementById('results');

form.addEventListener('submit', async e => {
  e.preventDefault();

  if (!mriInput.files[0]) {
    alert('Please upload an MRI scan first.');
    return;
  }

  // Show loading
  loadingOverlay.style.display = 'flex';
  results.style.display = 'none';
  animateLoadingSteps();

  const formData = new FormData(form);

  // Handle checkboxes properly
  ['smoking','family_history','cardiovascular','diabetes','depression',
   'head_injury','hypertension','memory_complaints','behavioral_problems',
   'confusion','disorientation','personality_changes','difficulty_tasks','forgetfulness'
  ].forEach(name => {
    if (!formData.has(name)) formData.set(name, '0');
  });

  try {
    const res  = await fetch('/analyze', { method: 'POST', body: formData });
    const data = await res.json();

    if (!data.success) throw new Error(data.error);

    await delay(400);
    loadingOverlay.style.display = 'none';
    renderResults(data);

  } catch (err) {
    loadingOverlay.style.display = 'none';
    alert('Analysis failed: ' + err.message);
  }
});

// ── Loading Animation ─────────────────────────────────────
function animateLoadingSteps() {
  const steps = ['ls1', 'ls2', 'ls3'];
  steps.forEach(id => {
    document.getElementById(id).className = 'loading-step';
  });
  document.getElementById('ls1').classList.add('active');

  setTimeout(() => {
    document.getElementById('ls1').classList.remove('active');
    document.getElementById('ls1').classList.add('done');
    document.getElementById('ls2').classList.add('active');
  }, 1200);

  setTimeout(() => {
    document.getElementById('ls2').classList.remove('active');
    document.getElementById('ls2').classList.add('done');
    document.getElementById('ls3').classList.add('active');
  }, 2400);
}

// ── Render Results ────────────────────────────────────────
function renderResults(data) {
  const { mri, risk, report } = data;

  // MRI card
  document.getElementById('mriStage').textContent = mri.stage;
  document.getElementById('mriStage').style.color = mri.color;
  document.getElementById('mriDesc').textContent  = mri.description;
  const confBadge = document.getElementById('mriConfidence');
  confBadge.textContent = mri.confidence + '% confidence';
  confBadge.style.background = hexToRgba(mri.color, 0.1);
  confBadge.style.color = mri.color;

  // Probability bars
  const probBars = document.getElementById('probBars');
  probBars.innerHTML = '';
  const classColors = {
    'NonDemented':      '#34C759',
    'VeryMildDemented': '#FF9F0A',
    'MildDemented':     '#FF6B35',
    'ModerateDemented': '#FF3B30'
  };
  const classNames = {
    'NonDemented':      'No Dementia',
    'VeryMildDemented': 'Very Mild',
    'MildDemented':     'Mild',
    'ModerateDemented': 'Moderate'
  };

  Object.entries(mri.all_probs)
    .sort((a, b) => b[1] - a[1])
    .forEach(([cls, pct]) => {
      const color = classColors[cls] || '#aaa';
      const div = document.createElement('div');
      div.className = 'prob-item';
      div.innerHTML = `
        <div class="prob-row">
          <span class="prob-name">${classNames[cls] || cls}</span>
          <span class="prob-pct" style="color:${color}">${pct.toFixed(1)}%</span>
        </div>
        <div class="prob-bar-bg">
          <div class="prob-bar-fill" style="width:0%; background:${color}" data-width="${pct}"></div>
        </div>`;
      probBars.appendChild(div);
    });

  // Risk card
  const riskLabel = document.getElementById('riskLabel');
  riskLabel.textContent = risk.label + ' Risk';
  riskLabel.style.background = hexToRgba(risk.color, 0.1);
  riskLabel.style.color = risk.color;

  document.getElementById('riskDesc').textContent =
    `${risk.probability}% probability of Alzheimer's progression based on clinical data analysis.`;

  // Gauge
  const gaugeArc   = document.getElementById('gaugeArc');
  const gaugeValue = document.getElementById('gaugeValue');
  gaugeValue.style.color = risk.color;
  const circumference = 251.2;
  const offset = circumference - (risk.probability / 100) * circumference;

  // Report
  document.getElementById('reportText').textContent = report;

  // Show results
  results.style.display = 'block';
  results.scrollIntoView({ behavior: 'smooth', block: 'start' });

  // Animate bars after render
  requestAnimationFrame(() => {
    setTimeout(() => {
      document.querySelectorAll('.prob-bar-fill').forEach(bar => {
        bar.style.width = bar.dataset.width + '%';
      });
      gaugeArc.style.strokeDashoffset = offset;
      animateCount(gaugeValue, 0, risk.probability, 1500, '%');
    }, 300);
  });
}

// ── Helpers ───────────────────────────────────────────────
function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1,3), 16);
  const g = parseInt(hex.slice(3,5), 16);
  const b = parseInt(hex.slice(5,7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

function animateCount(el, from, to, duration, suffix) {
  const start = performance.now();
  const update = (time) => {
    const progress = Math.min((time - start) / duration, 1);
    const ease     = 1 - Math.pow(1 - progress, 3);
    el.textContent = Math.round(from + (to - from) * ease) + suffix;
    if (progress < 1) requestAnimationFrame(update);
  };
  requestAnimationFrame(update);
}

function resetForm() {
  form.reset();
  uploadContent.style.display = 'flex';
  uploadPreview.style.display = 'none';
  results.style.display = 'none';
  window.scrollTo({ top: 0, behavior: 'smooth' });
}