// Shared data model + storage helpers used by the popup and the dashboard.

const MILESTONES = [
  { key: 'applied', label: 'Applied' },
  { key: 'responded', label: 'Got a response' },
  { key: 'interviewScheduled', label: 'Interview scheduled' },
  { key: 'interviewed', label: 'Interview attended' },
  { key: 'offer', label: 'Offer received' },
  { key: 'rejected', label: 'Rejected' },
  { key: 'withdrawn', label: 'Withdrawn' },
];

// Status is derived from the furthest milestone reached (closing states win).
const STATUSES = [
  { key: 'saved', label: 'Saved' },
  { key: 'applied', label: 'Applied' },
  { key: 'responded', label: 'Responded' },
  { key: 'interviewScheduled', label: 'Interview scheduled' },
  { key: 'interviewed', label: 'Interviewed' },
  { key: 'offer', label: 'Offer' },
  { key: 'rejected', label: 'Rejected' },
  { key: 'withdrawn', label: 'Withdrawn' },
];

function deriveStatus(job) {
  const m = job.milestones || {};
  for (const key of ['withdrawn', 'rejected', 'offer', 'interviewed', 'interviewScheduled', 'responded', 'applied']) {
    if (m[key]) return key;
  }
  return 'saved';
}

function statusLabel(key) {
  return (STATUSES.find((s) => s.key === key) || STATUSES[0]).label;
}

function today() {
  // Local date as YYYY-MM-DD.
  return new Date().toLocaleDateString('en-CA');
}

function uid() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

// Strip tracking noise so the same posting matches across visits.
function normalizeUrl(raw) {
  try {
    const u = new URL(raw);
    u.hash = '';
    const host = u.hostname.replace(/^www\./, '');

    // LinkedIn: search/collection pages carry the job in ?currentJobId=
    if (host.endsWith('linkedin.com')) {
      const id = u.searchParams.get('currentJobId') || (u.pathname.match(/\/jobs\/view\/(?:[^/]*?-)?(\d+)/) || [])[1];
      if (id) return `https://www.linkedin.com/jobs/view/${id}/`;
    }
    // Indeed: job is in ?jk= or ?vjk=
    if (host.includes('indeed.')) {
      const jk = u.searchParams.get('jk') || u.searchParams.get('vjk');
      if (jk) return `https://${u.hostname}/viewjob?jk=${jk}`;
    }

    const drop = [];
    u.searchParams.forEach((_, k) => {
      if (/^(utm_|gh_src|ref|refId|trackingId|source|src|lever-|fbclid|gclid)/i.test(k)) drop.push(k);
    });
    drop.forEach((k) => u.searchParams.delete(k));
    return u.toString();
  } catch {
    return raw || '';
  }
}

async function getJobs() {
  const { jobs } = await chrome.storage.local.get('jobs');
  return jobs || {};
}

async function getJobList() {
  return Object.values(await getJobs());
}

async function saveJob(job) {
  const jobs = await getJobs();
  const now = new Date().toISOString();
  const existing = job.id ? jobs[job.id] : null;
  const record = {
    milestones: {},
    ...existing,
    ...job,
    id: job.id || uid(),
    createdAt: existing?.createdAt || job.createdAt || now,
    updatedAt: now,
  };
  record.url = record.url ? normalizeUrl(record.url) : '';
  jobs[record.id] = record;
  await chrome.storage.local.set({ jobs });
  return record;
}

async function deleteJob(id) {
  const jobs = await getJobs();
  delete jobs[id];
  await chrome.storage.local.set({ jobs });
}

async function findJobByUrl(url) {
  const target = normalizeUrl(url);
  if (!target) return null;
  return (await getJobList()).find((j) => j.url && normalizeUrl(j.url) === target) || null;
}

// Days between a YYYY-MM-DD string and today.
function daysSince(dateStr) {
  if (!dateStr) return null;
  const [y, m, d] = dateStr.split('-').map(Number);
  const then = new Date(y, m - 1, d);
  const now = new Date();
  now.setHours(0, 0, 0, 0);
  return Math.round((now - then) / 86400000);
}
