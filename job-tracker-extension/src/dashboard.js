/* global MILESTONES, STATUSES, deriveStatus, statusLabel, today, escapeHtml, getJobs, getJobList,
   saveJob, deleteJob, daysSince */

const TABLE_MILESTONES = ['responded', 'interviewScheduled', 'interviewed', 'offer', 'rejected'];
const CLOSED = new Set(['rejected', 'withdrawn']);

const $ = (id) => document.getElementById(id);
const body = $('jobs-body');
const editor = $('editor');
const editForm = $('edit-form');

let jobs = [];
let editing = null; // job being edited in the dialog

// ---------- Filters (remembered per browser) ----------
const filters = { search: '', status: '', sort: 'applied-desc', hideClosed: false };
try { Object.assign(filters, JSON.parse(localStorage.getItem('filters') || '{}')); } catch { /* ignore */ }

$('status-filter').innerHTML += STATUSES.map((s) => `<option value="${s.key}">${escapeHtml(s.label)}</option>`).join('');
$('status-filter').value = filters.status;
$('sort').value = filters.sort;
$('hide-closed').checked = filters.hideClosed;
$('search').value = filters.search;

function persistFilters() {
  try { localStorage.setItem('filters', JSON.stringify(filters)); } catch { /* ignore */ }
}
$('search').addEventListener('input', (e) => { filters.search = e.target.value; persistFilters(); renderTable(); });
$('status-filter').addEventListener('change', (e) => { filters.status = e.target.value; persistFilters(); renderTable(); });
$('sort').addEventListener('change', (e) => { filters.sort = e.target.value; persistFilters(); renderTable(); });
$('hide-closed').addEventListener('change', (e) => { filters.hideClosed = e.target.checked; persistFilters(); renderTable(); });

// ---------- Rendering ----------
function renderStats() {
  const count = (k) => jobs.filter((j) => j.milestones?.[k]).length;
  const applied = count('applied');
  const responded = jobs.filter((j) => {
    const m = j.milestones || {};
    return m.responded || m.interviewScheduled || m.interviewed || m.offer || m.rejected;
  }).length;
  const interviewing = jobs.filter((j) => ['interviewScheduled', 'interviewed'].includes(deriveStatus(j))).length;
  const pct = (n) => (applied ? `${Math.round((n / applied) * 100)}% of applied` : '');
  const awaiting = jobs.filter((j) => deriveStatus(j) === 'applied').length;

  const cards = [
    ['Tracked', jobs.length, `${jobs.length - applied} saved, not applied`],
    ['Applied', applied, ''],
    ['Heard back', responded, pct(responded)],
    ['Awaiting reply', awaiting, ''],
    ['Interviewing now', interviewing, ''],
    ['Interviewed', count('interviewed'), pct(count('interviewed'))],
    ['Offers', count('offer'), pct(count('offer'))],
    ['Rejected', count('rejected'), ''],
  ];
  $('stats').innerHTML = cards.map(([l, n, sub]) =>
    `<div class="stat"><div class="n">${n}</div><div class="l">${l}</div>${sub ? `<div class="sub">${sub}</div>` : ''}</div>`
  ).join('');

  const now = new Date().toISOString().slice(0, 16);
  const upcoming = jobs
    .filter((j) => j.nextInterview && j.nextInterview >= now.slice(0, 10) && !CLOSED.has(deriveStatus(j)))
    .sort((a, b) => a.nextInterview.localeCompare(b.nextInterview));
  const up = $('upcoming');
  up.classList.toggle('hidden', !upcoming.length);
  up.innerHTML = upcoming.length
    ? `<h2>Upcoming interviews</h2><ul>${upcoming.map((j) =>
      `<li><strong>${escapeHtml(new Date(j.nextInterview).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }))}</strong> — ${escapeHtml(j.title)} at ${escapeHtml(j.company)}</li>`
    ).join('')}</ul>`
    : '';
}

function filteredJobs() {
  const q = filters.search.trim().toLowerCase();
  let list = jobs.filter((j) => {
    const st = deriveStatus(j);
    if (filters.status && st !== filters.status) return false;
    if (filters.hideClosed && CLOSED.has(st)) return false;
    if (q && ![j.title, j.company, j.location, j.notes, j.contact].join(' ').toLowerCase().includes(q)) return false;
    return true;
  });
  const appliedKey = (j) => j.milestones?.applied || j.createdAt.slice(0, 10);
  const sorters = {
    'applied-desc': (a, b) => appliedKey(b).localeCompare(appliedKey(a)),
    'applied-asc': (a, b) => appliedKey(a).localeCompare(appliedKey(b)),
    'updated-desc': (a, b) => b.updatedAt.localeCompare(a.updatedAt),
    'company-asc': (a, b) => (a.company || '').localeCompare(b.company || '', undefined, { sensitivity: 'base' }),
  };
  return list.sort(sorters[filters.sort] || sorters['applied-desc']);
}

function renderTable() {
  const list = filteredJobs();
  const empty = $('empty');
  if (!jobs.length) {
    empty.textContent = 'No applications yet. Open a job posting, click the extension icon and hit “Save application”.';
  } else if (!list.length) {
    empty.textContent = 'No applications match your filters.';
  }
  empty.classList.toggle('hidden', list.length > 0);
  $('jobs-table').querySelector('thead').classList.toggle('hidden', !list.length);

  body.innerHTML = list.map((j) => {
    const st = deriveStatus(j);
    const m = j.milestones || {};
    const ago = daysSince(m.applied);
    const waiting = st === 'applied' && ago >= 14;
    const titleHtml = j.url
      ? `<a href="${escapeHtml(j.url)}" target="_blank" rel="noopener">${escapeHtml(j.title)}</a>`
      : escapeHtml(j.title);
    const sub = [j.company, j.location, j.salary].filter(Boolean).map(escapeHtml).join(' · ');
    return `<tr data-id="${escapeHtml(j.id)}" class="${CLOSED.has(st) ? 'closed' : ''}">
      <td class="pos"><strong>${titleHtml}</strong><span class="sub">${sub}</span></td>
      <td class="applied">${m.applied
        ? `${escapeHtml(m.applied)}<span class="ago ${waiting ? 'stale' : ''}">${ago === 0 ? 'today' : `${ago}d ago`}${waiting ? ' · follow up?' : ''}</span>`
        : `<button class="link" data-mark-applied>Mark applied</button>`}</td>
      ${TABLE_MILESTONES.map((k) => `<td class="c"><input type="checkbox" data-key="${k}" ${m[k] ? 'checked' : ''}
        aria-label="${escapeHtml(MILESTONES.find((x) => x.key === k).label)}">${m[k] ? `<span class="d">${escapeHtml(m[k].slice(5))}</span>` : ''}</td>`).join('')}
      <td><span class="pill" data-status="${st}">${escapeHtml(statusLabel(st))}</span></td>
      <td><button data-edit>Edit</button></td>
    </tr>`;
  }).join('');
}

function render() {
  renderStats();
  renderTable();
}

async function reload() {
  jobs = await getJobList();
  render();
}

// ---------- Inline table actions ----------
body.addEventListener('change', async (e) => {
  const key = e.target.dataset.key;
  const id = e.target.closest('tr')?.dataset.id;
  const job = jobs.find((j) => j.id === id);
  if (!key || !job) return;
  const milestones = { ...job.milestones };
  if (e.target.checked) milestones[key] = today();
  else delete milestones[key];
  await saveJob({ ...job, milestones });
});

body.addEventListener('click', async (e) => {
  const id = e.target.closest('tr')?.dataset.id;
  const job = jobs.find((j) => j.id === id);
  if (!job) return;
  if (e.target.matches('[data-edit]')) openEditor(job);
  if (e.target.matches('[data-mark-applied]')) {
    await saveJob({ ...job, milestones: { ...job.milestones, applied: today() } });
  }
});

// ---------- Editor dialog ----------
function renderEditMilestones() {
  $('edit-milestones').innerHTML = MILESTONES.map(({ key, label }) => {
    const d = editing.milestones[key] || '';
    return `<div class="ms">
      <label><input type="checkbox" data-key="${key}" ${d ? 'checked' : ''}> ${escapeHtml(label)}</label>
      <input type="date" data-date="${key}" value="${escapeHtml(d)}" ${d ? '' : 'disabled'}>
    </div>`;
  }).join('');
}

$('edit-milestones').addEventListener('change', (e) => {
  const { key, date } = e.target.dataset;
  if (key) {
    if (e.target.checked) editing.milestones[key] = today();
    else delete editing.milestones[key];
    renderEditMilestones();
  } else if (date) {
    if (e.target.value) editing.milestones[date] = e.target.value;
  }
});

function openEditor(job) {
  editing = job
    ? { ...job, milestones: { ...job.milestones } }
    : { milestones: { applied: today() } };
  $('editor-title').textContent = job ? 'Edit application' : 'Add application';
  $('delete-btn').classList.toggle('hidden', !job);
  for (const el of editForm.elements) {
    if (el.name) el.value = editing[el.name] ?? '';
  }
  renderEditMilestones();
  editor.showModal();
}

editForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = Object.fromEntries(new FormData(editForm));
  const job = { ...editing, ...fd, milestones: editing.milestones };
  if (!job.id) delete job.id;
  await saveJob(job);
  editor.close();
});

$('cancel-btn').addEventListener('click', () => editor.close());
$('delete-btn').addEventListener('click', async () => {
  if (!editing?.id) return;
  if (!confirm(`Delete “${editing.title}” at ${editing.company}? This can’t be undone.`)) return;
  await deleteJob(editing.id);
  editor.close();
});
$('add-btn').addEventListener('click', () => openEditor(null));

// ---------- Import / export ----------
function download(filename, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const a = Object.assign(document.createElement('a'), { href: url, download: filename });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

$('export-csv').addEventListener('click', () => {
  const cols = [
    ['Position', (j) => j.title], ['Company', (j) => j.company], ['Location', (j) => j.location],
    ['Salary', (j) => j.salary], ['Status', (j) => statusLabel(deriveStatus(j))],
    ...MILESTONES.map((m) => [m.label, (j) => j.milestones?.[m.key] || '']),
    ['Next interview', (j) => j.nextInterview], ['Contact', (j) => j.contact],
    ['URL', (j) => j.url], ['Notes', (j) => j.notes],
  ];
  const esc = (v) => `"${String(v ?? '').replace(/"/g, '""')}"`;
  const rows = [cols.map(([h]) => esc(h)).join(',')]
    .concat(filteredJobs().map((j) => cols.map(([, f]) => esc(f(j))).join(',')));
  download(`job-applications-${today()}.csv`, '﻿' + rows.join('\r\n'), 'text/csv');
});

$('export-json').addEventListener('click', async () => {
  const data = { version: 1, exportedAt: new Date().toISOString(), jobs: await getJobs() };
  download(`job-tracker-backup-${today()}.json`, JSON.stringify(data, null, 2), 'application/json');
});

$('import-btn').addEventListener('click', () => $('import-file').click());
$('import-file').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  e.target.value = '';
  if (!file) return;
  try {
    const data = JSON.parse(await file.text());
    const incoming = data.jobs && typeof data.jobs === 'object' ? data.jobs : null;
    if (!incoming) throw new Error('No jobs found in file');
    const n = Object.keys(incoming).length;
    if (!confirm(`Restore ${n} application(s)? Entries with the same ID will be overwritten; others are kept.`)) return;
    const merged = { ...(await getJobs()), ...incoming };
    await chrome.storage.local.set({ jobs: merged });
  } catch (err) {
    alert(`Couldn’t import that file: ${err.message}`);
  }
});

// Keep in sync with saves from the popup (or another dashboard tab).
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === 'local' && changes.jobs) reload();
});

reload();
