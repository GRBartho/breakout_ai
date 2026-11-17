// Injected into the job posting tab via chrome.scripting.executeScript.
// MUST stay self-contained: it is serialized and run in the page, so it
// cannot reference anything outside its own body.
function extractJobInfo() {
  const clean = (s) => String(s || '').replace(/\s+/g, ' ').trim();
  const stripHtml = (html) => {
    const div = document.createElement('div');
    div.innerHTML = String(html || '');
    return clean(div.textContent);
  };
  const text = (...selectors) => {
    for (const sel of selectors) {
      try {
        const el = document.querySelector(sel);
        const t = el && clean(el.innerText || el.textContent);
        if (t) return t;
      } catch { /* bad selector on this page, ignore */ }
    }
    return '';
  };
  const meta = (name) =>
    clean(document.querySelector(`meta[property="${name}"], meta[name="${name}"]`)?.content);

  const out = {
    title: '', company: '', location: '', salary: '', employmentType: '',
    datePosted: '', description: '', url: location.href, source: location.hostname.replace(/^www\./, ''),
  };
  const fill = (data) => {
    for (const [k, v] of Object.entries(data)) if (!out[k] && clean(v)) out[k] = clean(v);
  };

  // ---- 1. schema.org JobPosting (JSON-LD) — used by most ATSs and job boards
  const findPosting = (node) => {
    if (!node || typeof node !== 'object') return null;
    if (Array.isArray(node)) {
      for (const n of node) { const r = findPosting(n); if (r) return r; }
      return null;
    }
    const t = node['@type'];
    if (t === 'JobPosting' || (Array.isArray(t) && t.includes('JobPosting'))) return node;
    if (node['@graph']) return findPosting(node['@graph']);
    return null;
  };
  let posting = null;
  for (const s of document.querySelectorAll('script[type="application/ld+json"]')) {
    try { posting = findPosting(JSON.parse(s.textContent)); } catch { /* malformed JSON-LD */ }
    if (posting) break;
  }
  if (posting) {
    const org = posting.hiringOrganization;
    const locs = [].concat(posting.jobLocation || []);
    const locStr = locs.map((l) => {
      const a = l?.address || {};
      if (typeof a === 'string') return a;
      const country = typeof a.addressCountry === 'object' ? a.addressCountry?.name : a.addressCountry;
      return [a.addressLocality, a.addressRegion, country].filter(Boolean).join(', ');
    }).filter(Boolean);
    const remote = [].concat(posting.jobLocationType || []).some((t) => /telecommute|remote/i.test(t));
    if (remote) locStr.unshift('Remote');

    let salary = '';
    const bs = posting.baseSalary || posting.estimatedSalary;
    const sal = Array.isArray(bs) ? bs[0] : bs;
    if (sal && typeof sal === 'object') {
      const v = sal.value || {};
      const cur = sal.currency || v.currency || '';
      const fmt = (n) => (n == null || n === '' ? '' : Number(n).toLocaleString());
      const range = typeof v === 'object'
        ? (v.minValue != null && v.maxValue != null && v.minValue !== v.maxValue
          ? `${fmt(v.minValue)} – ${fmt(v.maxValue)}`
          : fmt(v.value ?? v.minValue ?? v.maxValue))
        : fmt(v);
      const unit = (typeof v === 'object' && v.unitText) || sal.unitText || '';
      if (range) salary = [cur, range, unit && `/ ${String(unit).toLowerCase()}`].filter(Boolean).join(' ');
    }

    fill({
      title: stripHtml(posting.title),
      company: typeof org === 'string' ? org : org?.name,
      location: [...new Set(locStr)].join(' · '),
      salary,
      employmentType: [].concat(posting.employmentType || []).join(', ').replace(/_/g, ' ').toLowerCase(),
      datePosted: String(posting.datePosted || '').slice(0, 10),
      description: stripHtml(posting.description),
    });
  }

  // ---- 2. Site-specific selectors (fill whatever JSON-LD didn't give us)
  const host = location.hostname;
  const path = location.pathname.split('/').filter(Boolean);
  const titleCase = (slug) => clean(String(slug || '').replace(/[-_]+/g, ' ')).replace(/\b\w/g, (c) => c.toUpperCase());

  if (host.includes('linkedin.com')) {
    fill({
      title: text('.job-details-jobs-unified-top-card__job-title h1', '.job-details-jobs-unified-top-card__job-title',
        '.jobs-unified-top-card__job-title', '.top-card-layout__title', 'h1'),
      company: text('.job-details-jobs-unified-top-card__company-name', '.jobs-unified-top-card__company-name',
        '.topcard__org-name-link', '.top-card-layout__second-subline a'),
      location: text('.job-details-jobs-unified-top-card__primary-description-container .tvm__text',
        '.job-details-jobs-unified-top-card__bullet', '.topcard__flavor--bullet'),
      description: text('#job-details', '.jobs-description__content', '.show-more-less-html__markup'),
    });
  } else if (host.includes('indeed.')) {
    fill({
      title: text('[data-testid="jobsearch-JobInfoHeader-title"]', 'h1.jobsearch-JobInfoHeader-title', 'h1')
        .replace(/\s*-\s*job post$/i, ''),
      company: text('[data-testid="inlineHeader-companyName"]', '[data-company-name="true"]', '.jobsearch-CompanyInfoContainer a'),
      location: text('[data-testid="inlineHeader-companyLocation"]', '[data-testid="job-location"]', '#jobLocationText'),
      salary: text('#salaryInfoAndJobType span', '[data-testid="jobsearch-OtherJobDetailsContainer"] span'),
      description: text('#jobDescriptionText'),
    });
  } else if (host.includes('greenhouse.io')) {
    const boardSlug = host.startsWith('boards') || host.startsWith('job-boards') ? path[0] : '';
    fill({
      title: text('.job__title h1', '.app-title', 'h1.section-header', 'h1'),
      company: text('.company-name').replace(/^at\s+/i, '') || titleCase(boardSlug),
      location: text('.job__location', '.location'),
      description: text('.job__description', '#content'),
    });
  } else if (host.includes('lever.co')) {
    fill({
      title: text('.posting-headline h2', 'h2'),
      company: titleCase(path[0]),
      location: text('.posting-categories .location', '.posting-categories .sort-by-time'),
      employmentType: text('.posting-categories .commitment'),
      description: text('.section-wrapper.page-full-width', '[data-qa="job-description"]'),
    });
  } else if (host.includes('myworkdayjobs.com') || host.includes('workday')) {
    fill({
      title: text('[data-automation-id="jobPostingHeader"]', 'h2', 'h1'),
      company: titleCase(host.split('.')[0]),
      location: text('[data-automation-id="locations"] dd', '[data-automation-id="locations"]'),
      employmentType: text('[data-automation-id="time"] dd'),
      description: text('[data-automation-id="jobPostingDescription"]'),
    });
  } else if (host.includes('ashbyhq.com')) {
    fill({
      title: text('h1'),
      company: titleCase(path[0]),
      location: text('[class*="_section"] p', '[class*="location"]'),
      description: text('[class*="_descriptionText"]', '#overview'),
    });
  } else if (host.includes('glassdoor.')) {
    fill({
      title: text('[data-test="job-title"]', 'h1'),
      company: text('[data-test="employer-name"]', '[data-test="employerName"]').replace(/\d(\.\d)?\s*★?$/, ''),
      location: text('[data-test="location"]', '[data-test="emp-location"]'),
      salary: text('[data-test="detailSalary"]'),
      description: text('[class*="JobDetails_jobDescription"]', '.jobDescriptionContent'),
    });
  } else if (host.includes('smartrecruiters.com')) {
    fill({
      title: text('h1.job-title', 'h1'),
      company: text('[itemprop="hiringOrganization"] [itemprop="name"]') || titleCase(path[0]),
      location: text('[itemprop="jobLocation"]', '.job-location'),
      description: text('[itemprop="description"]'),
    });
  } else if (host.includes('wellfound.com') || host.includes('angel.co')) {
    fill({ title: text('h1'), company: text('a[href*="/company/"] h2', 'a[href*="/company/"]') });
  }

  // ---- 3. Generic fallbacks
  // "Senior Engineer at Acme | LinkedIn" / "Senior Engineer - Acme - Indeed.com"
  const docTitle = clean(meta('og:title') || document.title);
  const atMatch = docTitle.match(/^(.+?)\s+(?:at|@)\s+(.+?)(?:\s+[|\-–—]\s+.*)?$/i);
  if (atMatch) fill({ title: atMatch[1], company: atMatch[2] });
  fill({
    title: text('h1') || docTitle.split(/\s+[|\-–—]\s+/)[0],
    company: /linkedin|indeed|glassdoor|greenhouse|lever|workday|ziprecruiter|monster|dice|wellfound|ashby|smartrecruiters/i
      .test(meta('og:site_name')) ? '' : meta('og:site_name'),
    description: meta('og:description') || meta('description'),
  });

  if (out.description.length > 4000) out.description = out.description.slice(0, 4000) + '…';
  return out;
}
