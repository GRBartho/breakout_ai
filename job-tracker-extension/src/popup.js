/* global MILESTONES, deriveStatus, statusLabel, today, escapeHtml, saveJob, findJobByUrl,
   getJobList, normalizeUrl, extractJobInfo */

const form = document.getElementById('job-form');
const banner = document.getElementById('banner');
const milestonesEl = document.getElementById('milestones');
const statusPill = document.getElementById('status-pill');
const saveBtn = document.getElementById('save-btn');

// Holds data that has no visible input (description, dates per milestone, etc.).
let current = { milestones: {} };

function showBanner(msg, kind = '') {
  banner.textContent = msg;
  banner.className = `banner ${kind}`;
}

function renderMilestones() {
  milestonesEl.innerHTML = MILESTONES.map(({ key, label }) => {
    const date = current.milestones[key];
    return `<label>
      <input type="checkbox" data-key="${key}" ${date ? 'checked' : ''}>
      ${escapeHtml(label)}
      <span class="date">${date ? escapeHtml(date.slice(5)) : ''}</span>
    </label>`;
  }).join('');
  renderStatus();
}

function renderStatus() {
  const st = deriveStatus(current);
  statusPill.innerHTML = `<span class="pill" data-status="${st}">${escapeHtml(statusLabel(st))}</span>`;
}

milestonesEl.addEventListener('change', (e) => {
  const key = e.target.dataset.key;
  if (!key) return;
  if (e.target.checked) current.milestones[key] = today();
  else delete current.milestones[key];
  renderMilestones();
});

function fillForm(data) {
  for (const el of form.elements) {
    if (el.name && el.name in data) el.value = data[el.name] ?? '';
  }
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = Object.fromEntries(new FormData(form));
  const job = { ...current, ...fd, milestones: { ...current.milestones } };
  if (!job.id) delete job.id;
  saveBtn.disabled = true;
  try {
    current = await saveJob(job);
    fillForm(current);
    saveBtn.textContent = 'Update application';
    showBanner(`Saved “${current.title}” at ${current.company}.`, 'ok');
    renderRecent();
  } finally {
    saveBtn.disabled = false;
  }
});

document.getElementById('open-dashboard').addEventListener('click', () => {
  chrome.tabs.create({ url: chrome.runtime.getURL('dashboard.html') });
  window.close();
});

async function renderRecent() {
  const list = (await getJobList()).sort((a, b) => b.updatedAt.localeCompare(a.updatedAt)).slice(0, 5);
  const ul = document.getElementById('recent-list');
  if (!list.length) {
    ul.innerHTML = '<li class="muted">Nothing tracked yet. Open a job posting and hit save.</li>';
    return;
  }
  ul.innerHTML = list.map((j) => {
    const st = deriveStatus(j);
    return `<li>
      <div class="meta"><strong>${escapeHtml(j.title)}</strong><div>${escapeHtml(j.company)}${j.milestones?.applied ? ' · applied ' + escapeHtml(j.milestones.applied) : ''}</div></div>
      <span class="pill" data-status="${st}">${escapeHtml(statusLabel(st))}</span>
    </li>`;
  }).join('');
}

async function init() {
  banner.className = 'banner hidden';
  renderRecent();

  let tab;
  try {
    [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  } catch { /* no tab access */ }

  const url = tab?.url || '';
  const existing = url ? await findJobByUrl(url) : null;

  if (existing) {
    current = { ...existing, milestones: { ...existing.milestones } };
    fillForm(current);
    saveBtn.textContent = 'Update application';
    showBanner('You already track this job — update its progress below.', 'ok');
    renderMilestones();
    return;
  }

  // New job: pre-check "Applied" (most people click after applying) and scrape the page.
  current = { milestones: { applied: today() } };
  renderMilestones();

  if (!/^https?:/.test(url)) {
    showBanner('Open a job posting to auto-fill, or enter the details by hand.');
    return;
  }

  fillForm({ url: normalizeUrl(url) });
  try {
    const [{ result }] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: extractJobInfo,
    });
    if (result) {
      const { title, company, location, salary, employmentType, datePosted, description, source } = result;
      current = { ...current, employmentType, datePosted, description, source };
      fillForm({ title, company, location, salary });
      if (!title || !company) showBanner('Couldn’t detect everything on this page — please fill in the blanks.');
    }
  } catch (err) {
    showBanner('Can’t read this page (browser pages and the Web Store are off-limits). Enter details manually.');
    console.warn('Extraction failed', err);
  }
}

init();
