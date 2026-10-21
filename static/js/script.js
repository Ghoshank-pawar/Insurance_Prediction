/**
 * InsurePredict AI - Client Interactive Controller
 * Handles client-side validation, asynchronous model inference,
 * Chart.js comparison rendering, and BMI helper calculator.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const form = document.getElementById('insurance-form');
  const btnSubmit = document.getElementById('btn-submit-predict');
  const btnReset = document.getElementById('btn-reset');
  const btnRecalculate = document.getElementById('btn-recalculate');
  const formErrorAlert = document.getElementById('form-error-alert');
  const formErrorMessage = document.getElementById('form-error-message');

  const emptyState = document.getElementById('results-empty-state');
  const resultCard = document.getElementById('prediction-result-card');

  // Result display elements
  const resAmount = document.getElementById('res-amount');
  const resCurrencyTag = document.getElementById('res-currency-tag');
  const resTimestamp = document.getElementById('res-timestamp');
  const chipAge = document.getElementById('chip-age');
  const chipSex = document.getElementById('chip-sex');
  const chipBmi = document.getElementById('chip-bmi');
  const chipSmoker = document.getElementById('chip-smoker');
  const chipChildren = document.getElementById('chip-children');
  const chipRegion = document.getElementById('chip-region');

  // Engineered terms elements
  const engAgeBmi = document.getElementById('eng-age-bmi');
  const engSmokerBmi = document.getElementById('eng-smoker-bmi');
  const engAgeSq = document.getElementById('eng-age-sq');
  const engBmiSq = document.getElementById('eng-bmi-sq');

  // BMI Helper elements
  const btnToggleBmi = document.getElementById('btn-toggle-bmi-calc');
  const bmiDrawer = document.getElementById('bmi-helper-drawer');
  const btnCloseBmi = document.getElementById('btn-close-bmi');
  const btnApplyBmi = document.getElementById('btn-apply-bmi');
  const calcHeight = document.getElementById('calc-height');
  const calcWeight = document.getElementById('calc-weight');
  const inputBmi = document.getElementById('bmi');

  // Chart instance holder
  let comparisonChart = null;

  // -------------------------------------------------------------
  // 1. BMI Helper Logic
  // -------------------------------------------------------------
  if (btnToggleBmi && bmiDrawer) {
    btnToggleBmi.addEventListener('click', () => {
      const isVisible = bmiDrawer.style.display !== 'none';
      bmiDrawer.style.display = isVisible ? 'none' : 'block';
    });

    btnCloseBmi.addEventListener('click', () => {
      bmiDrawer.style.display = 'none';
    });

    btnApplyBmi.addEventListener('click', () => {
      const hCm = parseFloat(calcHeight.value);
      const wKg = parseFloat(calcWeight.value);

      if (!hCm || hCm <= 0 || !wKg || wKg <= 0) {
        alert('Please enter valid positive numbers for height and weight.');
        return;
      }

      const hM = hCm / 100.0;
      const computedBmi = (wKg / (hM * hM)).toFixed(2);
      inputBmi.value = computedBmi;
      clearFieldError('bmi');
      bmiDrawer.style.display = 'none';
    });
  }

  // -------------------------------------------------------------
  // 2. Client-side Validation Functions
  // -------------------------------------------------------------
  function clearAllErrors() {
    formErrorAlert.style.display = 'none';
    const invalidInputs = form.querySelectorAll('.is-invalid');
    invalidInputs.forEach(el => el.classList.remove('is-invalid'));
    const errorSpans = form.querySelectorAll('.field-error');
    errorSpans.forEach(el => (el.textContent = ''));
  }

  function showFieldError(fieldId, message) {
    const field = document.getElementById(fieldId);
    if (field) {
      field.classList.add('is-invalid');
    }
    const errorSpan = document.getElementById(`error-${fieldId}`);
    if (errorSpan) {
      errorSpan.textContent = message;
    }
  }

  function clearFieldError(fieldId) {
    const field = document.getElementById(fieldId);
    if (field) {
      field.classList.remove('is-invalid');
    }
    const errorSpan = document.getElementById(`error-${fieldId}`);
    if (errorSpan) {
      errorSpan.textContent = '';
    }
  }

  // Clear errors dynamically on input
  ['age', 'bmi', 'children', 'sex', 'region'].forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener('input', () => clearFieldError(id));
      el.addEventListener('change', () => clearFieldError(id));
    }
  });

  function validateClientSide(payload) {
    const errors = {};

    // Age
    const age = parseFloat(payload.age);
    if (isNaN(age) || !Number.isInteger(age) || age < 1 || age > 120) {
      errors.age = 'Enter a valid age between 1 and 120.';
    }

    // BMI
    const bmi = parseFloat(payload.bmi);
    if (isNaN(bmi) || bmi < 10.0 || bmi > 75.0) {
      errors.bmi = 'Enter a valid BMI between 10.0 and 75.0.';
    }

    // Dependents
    const children = parseFloat(payload.children);
    if (isNaN(children) || !Number.isInteger(children) || children < 0 || children > 20) {
      errors.children = 'Enter dependents between 0 and 20.';
    }

    // Sex
    if (!payload.sex || payload.sex === '') {
      errors.sex = 'Please select biological sex.';
    }

    // Region
    if (!payload.region || payload.region === '') {
      errors.region = 'Please select a residential region.';
    }

    return errors;
  }

  // -------------------------------------------------------------
  // 3. Chart Rendering
  // -------------------------------------------------------------
  function renderChart(predictionAmount, currencySymbol) {
    const ctx = document.getElementById('predictionComparisonChart');
    if (!ctx) return;

    if (comparisonChart) {
      comparisonChart.destroy();
    }

    // Baseline medical insurance benchmarks from US health insurance datasets
    const benchmarkHealthy = 3500;
    const benchmarkAverage = 13200;
    const benchmarkHighRisk = 39000;

    comparisonChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: [
          'Healthy Baseline',
          'National Average',
          'Your Estimate',
          'High Risk Tier'
        ],
        datasets: [
          {
            label: 'Annual Cost (' + currencySymbol + ')',
            data: [
              benchmarkHealthy,
              benchmarkAverage,
              predictionAmount,
              benchmarkHighRisk
            ],
            backgroundColor: [
              'rgba(16, 185, 129, 0.65)',
              'rgba(14, 165, 233, 0.65)',
              'rgba(0, 168, 150, 0.95)',
              'rgba(239, 68, 68, 0.65)'
            ],
            borderColor: [
              '#10b981',
              '#0ea5e9',
              '#007f73',
              '#ef4444'
            ],
            borderWidth: 1.5,
            borderRadius: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function (context) {
                return currencySymbol + context.parsed.y.toLocaleString('en-US', { minimumFractionDigits: 2 });
              }
            }
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            ticks: {
              callback: function (val) {
                return currencySymbol + (val >= 1000 ? val / 1000 + 'k' : val);
              },
              font: { family: "'Space Grotesk', monospace", size: 10 }
            },
            grid: { color: '#f1f5f9' }
          },
          x: {
            grid: { display: false },
            ticks: { font: { family: "'Plus Jakarta Sans', sans-serif", size: 10 } }
          }
        }
      }
    });
  }

  // -------------------------------------------------------------
  // 4. Form Submission Handling
  // -------------------------------------------------------------
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    clearAllErrors();

    // Collect data
    const formData = new FormData(form);
    const payload = {
      age: formData.get('age'),
      sex: formData.get('sex'),
      bmi: formData.get('bmi'),
      children: formData.get('children'),
      smoker: formData.get('smoker'),
      region: formData.get('region')
    };

    // Client-side validation check
    const clientErrors = validateClientSide(payload);
    if (Object.keys(clientErrors).length > 0) {
      for (const [fld, msg] of Object.entries(clientErrors)) {
        showFieldError(fld, msg);
      }
      formErrorAlert.style.display = 'flex';
      formErrorMessage.textContent = 'Please correct the highlighted fields before submitting.';
      return;
    }

    // Toggle button loading state
    const btnText = btnSubmit.querySelector('.btn-text');
    const btnSpinner = btnSubmit.querySelector('.btn-spinner');
    btnSubmit.disabled = true;
    if (btnText) btnText.style.display = 'none';
    if (btnSpinner) btnSpinner.style.display = 'inline-flex';

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Requested-With': 'XMLHttpRequest'
        },
        body: JSON.stringify(payload)
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        if (data.validation_errors) {
          for (const [fld, msg] of Object.entries(data.validation_errors)) {
            showFieldError(fld, msg);
          }
          formErrorAlert.style.display = 'flex';
          formErrorMessage.textContent = 'Server validation failed on the submitted values.';
        } else {
          formErrorAlert.style.display = 'flex';
          formErrorMessage.innerHTML = `<strong>Error:</strong> ${data.error || 'Server encountered an unexpected error.'}`;
        }
        return;
      }

      // Success: populate results
      emptyState.style.display = 'none';
      resultCard.style.display = 'block';

      resAmount.textContent = data.prediction_formatted;
      resCurrencyTag.textContent = `${data.currency} / Year`;
      resTimestamp.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

      // Profile chips
      chipAge.innerHTML = `Age: <strong>${data.user_inputs.age} yrs</strong>`;
      chipSex.innerHTML = `Sex: <strong>${data.user_inputs.sex}</strong>`;
      chipBmi.innerHTML = `BMI: <strong>${data.user_inputs.bmi}</strong> (${data.user_inputs.bmi_category})`;
      chipSmoker.innerHTML = `Smoker: <strong>${data.user_inputs.smoker}</strong>`;
      chipChildren.innerHTML = `Dependents: <strong>${data.user_inputs.children}</strong>`;
      chipRegion.innerHTML = `Region: <strong>${data.user_inputs.region}</strong>`;

      // Engineered values
      engAgeBmi.textContent = data.engineered_features.age_bmi;
      engSmokerBmi.textContent = data.engineered_features.smoker_bmi;
      engAgeSq.textContent = data.engineered_features.age_squared;
      engBmiSq.textContent = data.engineered_features.bmi_squared;

      // Render Comparison Chart
      renderChart(data.prediction, data.currency_symbol || '$');

      // Smooth scroll into results on mobile
      if (window.innerWidth < 1024) {
        resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }

    } catch (err) {
      console.error('Fetch error:', err);
      formErrorAlert.style.display = 'flex';
      formErrorMessage.textContent = 'Network or connection error. Please ensure the Flask server is running.';
    } finally {
      btnSubmit.disabled = false;
      if (btnText) btnText.style.display = 'inline';
      if (btnSpinner) btnSpinner.style.display = 'none';
    }
  });

  // -------------------------------------------------------------
  // 5. Reset & Recalculate
  // -------------------------------------------------------------
  btnReset.addEventListener('click', () => {
    form.reset();
    clearAllErrors();
    emptyState.style.display = 'flex';
    resultCard.style.display = 'none';
    if (comparisonChart) {
      comparisonChart.destroy();
      comparisonChart = null;
    }
  });

  if (btnRecalculate) {
    btnRecalculate.addEventListener('click', () => {
      document.getElementById('age').focus();
      document.getElementById('predict-form-section').scrollIntoView({ behavior: 'smooth' });
    });
  }
});
