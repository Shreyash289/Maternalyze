let probChart = null;
let featureChart = null;

function displayOutputAsTable(resultObj, targetDivId) {
  if (!resultObj || typeof resultObj !== 'object') {
    document.getElementById(targetDivId).textContent = 'No results available.';
    return;
  }

  let html = '<table class="card-output-table"><tbody>';

  for (const key in resultObj) {
    if (!resultObj.hasOwnProperty(key)) continue;
    const value = resultObj[key];
    // Human-readable label
    const label = key
      .replace(/([A-Z])/g, ' $1')
      .replace(/_/g, ' ')
      .replace(/\b\w/g, c => c.toUpperCase())
      .trim();

    if (Array.isArray(value)) {
      html += `<tr><th>${label}</th><td><ul class="precautions-list">`;
      value.forEach(item => {
        html += `<li>${item}</li>`;
      });
      html += "</ul></td></tr>";
    } else if (typeof value === 'object' && value !== null) {
      html += `<tr><th>${label}</th><td><pre>${JSON.stringify(value, null, 2)}</pre></td></tr>`;
    } else {
      html += `<tr><th>${label}</th><td>${value}</td></tr>`;
    }
  }

  html += '</tbody></table>';
  document.getElementById(targetDivId).innerHTML = html;
}

const gdmForm = document.getElementById('gdmForm');
if (gdmForm) {
  gdmForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());

    Object.keys(data).forEach(k => {
      if (!isNaN(data[k]) && data[k].toString().trim() !== '') data[k] = Number(data[k]);
    });

    try {
      const res = await fetch(`${API_BASE_URL}/predict_gdm`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      const json = await res.json();

      displayOutputAsTable(json, 'result');

      if (json?.Probability !== undefined) {
        const ctx = document.getElementById('probChart')?.getContext('2d');
        if (probChart) probChart.destroy();
        if (ctx) {
          probChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
              labels: ['Risk', 'No Risk'],
              datasets: [{
                data: [json.Probability * 100, 100 - json.Probability * 100],
                backgroundColor: ['#ff6384', '#36a2eb'],
              }]
            },
            options: {
              responsive: true,
              plugins: { legend: { position: 'bottom' } }
            }
          });
        }
      }
    } catch (error) {
      document.getElementById('result').textContent = `Error: ${error.message}`;
    }
  });
}

const childForm = document.getElementById('childForm');
if (childForm) {
  childForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());

    Object.keys(data).forEach(k => {
      if (!isNaN(data[k]) && data[k].toString().trim() !== '') data[k] = Number(data[k]);
    });

    try {
      const res = await fetch(`${API_BASE_URL}/predict_child`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      const json = await res.json();

      displayOutputAsTable(json, 'result');

      const features = json?.TopFeatureImportances || {};
      const labels = Object.keys(features);
      const values = Object.values(features);

      const ctx = document.getElementById('featureChart')?.getContext('2d');
      if (featureChart) featureChart.destroy();
      if (ctx) {
        featureChart = new Chart(ctx, {
          type: 'bar',
          data: {
            labels: labels,
            datasets: [{
              label: 'Feature Importance Gain',
              data: values,
              backgroundColor: '#36a2eb'
            }]
          },
          options: {
            responsive: true,
            scales: { y: { beginAtZero: true } }
          }
        });
      }
    } catch (error) {
      document.getElementById('result').textContent = `Error: ${error.message}`;
    }
  });
}