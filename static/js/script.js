const $ = (s) => document.querySelector(s);
const form = $("#predict-form");
const btn = $("#predict-btn");
const money = (n) => "$" + Math.round(n).toLocaleString("en-US");

function clearErrors() {
  form.querySelectorAll(".error").forEach((e) => (e.textContent = ""));
  form.querySelectorAll(".invalid").forEach((i) => i.classList.remove("invalid"));
}

function showFieldErrors(errors) {
  for (const [name, msg] of Object.entries(errors)) {
    const el = form.querySelector(`[data-error-for="${name}"]`);
    if (el) el.textContent = msg;
    if (form.elements[name]) form.elements[name].classList.add("invalid");
  }
}

function showResult(price, message) {
  $("#result-placeholder").hidden = true;
  $("#result-ok").hidden = message !== undefined;
  $("#result-error").hidden = message === undefined;
  if (message !== undefined) $("#result-error").textContent = message;
  else $("#price").textContent = price;
}

function validateClient(values) {
  const errors = {};
  form.querySelectorAll("input").forEach((input) => {
    const raw = values[input.name];
    const n = Number(raw);
    const min = Number(input.min), max = Number(input.max);
    if (input.validity.badInput) errors[input.name] = "Enter one number only, like -118.5.";
    else if (raw === "") errors[input.name] = "This field is required.";
    else if (!Number.isFinite(n)) errors[input.name] = "Enter a valid number.";
    else if (n < min || n > max) errors[input.name] = `Must be between ${min} and ${max}.`;
  });
  if (!values.ocean_proximity) errors.ocean_proximity = "Please choose an option.";
  if (!errors.total_rooms && !errors.total_bedrooms &&
      Number(values.total_bedrooms) > Number(values.total_rooms)) {
    errors.total_bedrooms = "Bedrooms cannot be more than total rooms.";
  }
  return errors;
}

form?.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearErrors();
  const values = Object.fromEntries(new FormData(form));
  const errors = validateClient(values);
  if (Object.keys(errors).length) {
    showFieldErrors(errors);
    showResult(null, "Please fix the highlighted inputs.");
    return;
  }
  btn.disabled = true;
  btn.textContent = "Predicting...";
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values),
    });
    const data = await res.json().catch(() => ({}));
    if (res.ok) {
      showResult(money(data.predicted_price));
    } else {
      if (data.fields) showFieldErrors(data.fields);
      showResult(null, data.error || "Something went wrong. Please try again.");
    }
  } catch (err) {
    showResult(null, "Cannot reach the server. Is app.py still running?");
  } finally {
    btn.disabled = false;
    btn.textContent = "Predict Price";
  }
});

$("#example-btn")?.addEventListener("click", (e) => {
  clearErrors();
  const example = JSON.parse(e.currentTarget.dataset.example);
  for (const [name, value] of Object.entries(example)) form.elements[name].value = value;
});

$("#reset-btn")?.addEventListener("click", () => {
  clearErrors();
  $("#result-ok").hidden = true;
  $("#result-error").hidden = true;
  $("#result-placeholder").hidden = false;
});

function cell(row, text) {
  const td = document.createElement("td");
  td.textContent = text;
  row.appendChild(td);
  return td;
}

async function loadMetrics() {
  const body = $("#metrics-body");
  try {
    const res = await fetch("/api/metrics");
    if (!res.ok) throw new Error("no metrics");
    const m = await res.json();
    body.replaceChildren();
    for (const [name, r] of Object.entries(m.models)) {
      const tr = document.createElement("tr");
      const first = cell(tr, name);
      if (name === m.best_model) {
        tr.className = "best";
        const badge = document.createElement("span");
        badge.className = "badge";
        badge.textContent = "Best";
        first.appendChild(badge);
      }
      cell(tr, Math.round(r.test.mae).toLocaleString("en-US"));
      cell(tr, Math.round(r.test.rmse).toLocaleString("en-US"));
      cell(tr, r.test.r2.toFixed(3));
      cell(tr, Math.round(r.cv_rmse).toLocaleString("en-US"));
      body.appendChild(tr);
    }
    const d = m.dataset;
    $("#chips").replaceChildren(...[
      `${d.rows.toLocaleString("en-US")} districts`,
      `${d.train_rows.toLocaleString("en-US")} train`,
      `${d.test_rows.toLocaleString("en-US")} test`,
      `Best: ${m.best_model}`,
    ].map((t) => Object.assign(document.createElement("span"), { textContent: t })));
    $("#metrics-note").textContent =
      "Best model chosen by " + m.selection_criterion + ". Trained " + m.trained_at.slice(0, 10) + ".";
  } catch (err) {
    body.innerHTML = '<tr><td colspan="5">Metrics not available. Run train_model.py first.</td></tr>';
  }
}

loadMetrics();

// ===== 3D tilt on cards and scroll-in animation =====
document.querySelectorAll(".grid-3 .card, figure.card, #contact .card").forEach((el) => {
  el.classList.add("tilt");
  el.addEventListener("mousemove", (e) => {
    const r = el.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width - 0.5;
    const y = (e.clientY - r.top) / r.height - 0.5;
    el.style.transform = `perspective(700px) rotateX(${-y * 10}deg) rotateY(${x * 10}deg) translateZ(6px)`;
  });
  el.addEventListener("mouseleave", () => (el.style.transform = ""));
});

const sections = document.querySelectorAll(".section, .alt");
sections.forEach((s) => s.classList.add("reveal"));
if ("IntersectionObserver" in window) {
  const io = new IntersectionObserver((entries) => {
    entries.forEach((en) => { if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); } });
  }, { threshold: 0.08 });
  sections.forEach((s) => io.observe(s));
} else {
  sections.forEach((s) => s.classList.add("in"));
}
