# Job Application Tracker (Chrome extension)

Save job postings in one click and keep track of where every application stands.

## Install (unpacked)

1. Open `chrome://extensions` in Chrome (or any Chromium browser).
2. Turn on **Developer mode** (top right).
3. Click **Load unpacked** and select this `job-tracker-extension` folder.
4. Pin the extension from the puzzle-piece menu so the icon is always visible.

## Using it

**Save a job.** Open a job posting and click the extension icon (or press `Alt+Shift+J`).
The popup fills in position, company, location, salary and URL from the page. Check or fix
the fields, then click **Save application**. "Applied" is ticked with today's date by default.
Untick it if you're only saving the job for later.

**Update progress.** Open the popup on a job you already saved and it loads that record, so you
can tick off steps. Or open **All applications →** for the dashboard, where you can:

- tick **Response / Interview scheduled / Interviewed / Offer / Rejected** right in the table
  (each tick records today's date)
- click **Edit** to change any date, add a recruiter contact, notes, or the **next interview**
  date and time (upcoming interviews are listed at the top)
- search, filter by status, sort, and hide closed applications
- see stats such as response rate and how many applications are waiting on a reply.
  Applications with no reply after 14 days get a "follow up?" hint.
- **Export CSV** for a spreadsheet, or **Backup** / **Restore** your data as JSON

The status is set by the furthest step you've reached:
Saved → Applied → Responded → Interview scheduled → Interviewed → Offer, or Rejected / Withdrawn.

## How it reads job pages

1. **schema.org `JobPosting` data** (JSON-LD). Most job boards and applicant tracking systems
   publish this for Google Jobs, so it's the most reliable source.
2. **Site-specific selectors** for LinkedIn, Indeed, Greenhouse, Lever, Workday, Ashby,
   Glassdoor, SmartRecruiters and Wellfound.
3. **Generic fallbacks**: the page `<h1>`, and titles like "Engineer at Acme | Site".

URLs are normalized so a posting is recognized again later. Tracking parameters are removed, and
LinkedIn `currentJobId` and Indeed `jk`/`vjk` links are reduced to the canonical job URL.

Job boards change their markup often. If a site stops auto-filling, update its selectors in
`src/extractor.js`. You can always type the details in by hand.

## Privacy

Everything is stored locally in `chrome.storage.local`. Nothing is sent anywhere. The extension
only reads a page when you click its icon (`activeTab`), and it needs no access to all sites.

## Files

| File | Purpose |
| --- | --- |
| `manifest.json` | Manifest V3 config |
| `popup.html`, `src/popup.js` | Popup: auto-fill and save the current job, quick progress ticks |
| `dashboard.html`, `src/dashboard.js` | Full list, stats, editor, import/export |
| `src/extractor.js` | Function injected into the job page to read the posting |
| `src/shared.js` | Data model, status logic, storage helpers |
| `src/*.css` | Styles (light and dark mode) |
