document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const form = document.getElementById('prediction-form');
  const ageInput = document.getElementById('age');
  const ageVal = document.getElementById('age-val');
  
  const bmiInput = document.getElementById('bmi');
  const bmiVal = document.getElementById('bmi-val');
  const bmiStatus = document.getElementById('bmi-status');
  
  const childrenInput = document.getElementById('children');
  const childrenVal = document.getElementById('children-val');
  
  const btnSubmit = document.getElementById('btn-submit');
  const btnText = document.getElementById('btn-text');
  const spinner = document.getElementById('spinner');
  
  const costDisplay = document.getElementById('predicted-cost');
  const riskBadge = document.getElementById('risk-badge');
  const percentileFill = document.getElementById('percentile-fill');
  const percentileText = document.getElementById('percentile-text');
  
  const riskList = document.getElementById('risk-list');
  const recsList = document.getElementById('recs-list');
  const modelNameText = document.getElementById('model-name-text');
  
  const btnModelInfo = document.getElementById('btn-model-info');
  const modalOverlay = document.getElementById('modal-overlay');
  const btnCloseModal = document.getElementById('btn-close-modal');
  const modalR2 = document.getElementById('modal-r2');
  const modalMae = document.getElementById('modal-mae');
  const modalRmse = document.getElementById('modal-rmse');
  const modalName = document.getElementById('modal-name');
  const modalParams = document.getElementById('modal-params');

  // Input Slider Synchronizers
  ageInput.addEventListener('input', (e) => {
    ageVal.textContent = e.target.value;
  });

  childrenInput.addEventListener('input', (e) => {
    childrenVal.textContent = e.target.value;
  });

  bmiInput.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value).toFixed(1);
    bmiVal.textContent = val;
    
    // Update BMI status pill
    if (val < 18.5) {
      bmiStatus.textContent = "Underweight";
      bmiStatus.className = "bmi-indicator bmi-overweight";
    } else if (val < 25.0) {
      bmiStatus.textContent = "Normal";
      bmiStatus.className = "bmi-indicator bmi-normal";
    } else if (val < 30.0) {
      bmiStatus.textContent = "Overweight";
      bmiStatus.className = "bmi-indicator bmi-overweight";
    } else {
      bmiStatus.textContent = "Obese";
      bmiStatus.className = "bmi-indicator bmi-obese";
    }
  });

  // Form Submission
  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Collect Input values
    const sex = document.querySelector('input[name="sex"]:checked').value;
    const smoker = document.querySelector('input[name="smoker"]:checked').value;
    const region = document.getElementById('region').value;

    const payload = {
      age: parseInt(ageInput.value),
      sex: sex,
      bmi: parseFloat(bmiInput.value),
      children: parseInt(childrenInput.value),
      smoker: smoker,
      region: region
    };

    // UI Loading State
    btnSubmit.disabled = true;
    spinner.style.display = 'inline-block';
    btnText.textContent = 'Calculating...';

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Prediction failed');
      }

      const result = await response.json();
      updateDashboard(result);

    } catch (err) {
      alert(`Error: ${err.message}`);
    } finally {
      btnSubmit.disabled = false;
      spinner.style.display = 'none';
      btnText.textContent = 'Predict Healthcare Cost';
    }
  });

  // Dashboard Renderer
  function updateDashboard(data) {
    // 1. Animate Cost Counter
    animateCounter(costDisplay, data.predicted_cost);

    // 2. Risk Level Badge
    riskBadge.textContent = data.risk_level;
    riskBadge.className = `badge-pill ${data.risk_badge_color}`;

    // 3. Percentile Progress
    percentileFill.style.width = `${data.cost_percentile}%`;
    percentileText.textContent = `Higher than ${data.cost_percentile}% of population`;

    // 4. Model Used
    modelNameText.textContent = `${data.model_used} (R² = ${data.model_r2_score})`;

    // 5. Risk Factors Breakdown
    riskList.innerHTML = '';
    data.top_risk_factors.forEach(item => {
      const div = document.createElement('div');
      div.className = `risk-item ${item.severity}`;
      div.innerHTML = `
        <div class="risk-icon">
          ${item.severity === 'critical' || item.severity === 'high' ? '⚠️' : item.severity === 'medium' ? '⚡' : '✓'}
        </div>
        <div class="risk-details">
          <h4>${item.factor} <span style="font-weight: normal; color: var(--text-secondary);">(${item.impact})</span></h4>
          <p>${item.description}</p>
        </div>
      `;
      riskList.appendChild(div);
    });

    // 6. Health Recommendations
    recsList.innerHTML = '';
    data.health_recommendations.forEach(rec => {
      const li = document.createElement('li');
      li.textContent = rec;
      recsList.appendChild(li);
    });
  }

  // Smooth Number Counter Animation
  function animateCounter(element, targetVal) {
    let startVal = 0;
    const duration = 800; // ms
    const startTime = performance.now();

    function update(now) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      const current = startVal + (targetVal - startVal) * (1 - Math.pow(1 - progress, 3));

      element.textContent = `$${current.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

      if (progress < 1) {
        requestAnimationFrame(update);
      }
    }

    requestAnimationFrame(update);
  }

  // Modal Model Info Drawer
  btnModelInfo.addEventListener('click', async () => {
    try {
      const res = await fetch('/model-info');
      if (!res.ok) throw new Error('Failed to fetch model info');
      const info = await res.json();

      modalName.textContent = info.model_name;
      modalR2.textContent = info.metrics.r2 ?? 'N/A';
      modalMae.textContent = info.metrics.mae ? `$${info.metrics.mae}` : 'N/A';
      modalRmse.textContent = info.metrics.rmse ? `$${info.metrics.rmse}` : 'N/A';

      modalParams.textContent = JSON.stringify(info.best_params, null, 2);
      modalOverlay.classList.add('active');
    } catch (err) {
      alert(`Model Info Error: ${err.message}`);
    }
  });

  btnCloseModal.addEventListener('click', () => {
    modalOverlay.classList.remove('active');
  });

  modalOverlay.addEventListener('click', (e) => {
    if (e.target === modalOverlay) {
      modalOverlay.classList.remove('active');
    }
  });

  // Initial Calculation on Load
  form.dispatchEvent(new Event('submit'));
});
