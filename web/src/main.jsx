import { lazy, Suspense, useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { modalities, request, useResource } from './api'
import './styles.css'

const Workspace = lazy(() => import('./Workspace'))

const pageTitles = {
  home: 'Overview',
  workspace: 'Case workspace',
  research: 'Research',
  figures: 'Figures',
}
const loadFigures = signal => request('/figures', { signal })

function currentPage() {
  const page = window.location.hash.slice(1)
  return Object.hasOwn(pageTitles, page) ? page : 'home'
}

function Header({ page }) {
  return (
    <header className="header">
      <div className="shell navigation">
        <a className="wordmark" href="#home" aria-label="CancerFusion AI home"><span aria-hidden="true" className="brand-mark"><i /><i /><i /><i /></span>CancerFusion<span> AI</span></a>
        <nav aria-label="Main navigation">
          {['home', 'research', 'figures'].map(key => <a key={key} href={'#' + key} aria-current={page === key ? 'page' : undefined}>{pageTitles[key]}</a>)}
        </nav>
        <a className="button primary nav-workspace" href="#workspace" aria-current={page === 'workspace' ? 'page' : undefined}>Open workspace <span aria-hidden="true">↗</span></a>
      </div>
    </header>
  )
}

function FlowDiagram() {
  return (
    <div className="flow-diagram" role="img" aria-label="Research prototype architecture with pathology, radiology, omics, and clinical feature branches. Four-input survival performance has not been validated.">
      <div className="flow-heading"><span>Research prototype architecture</span><code>PHASE 2</code></div>
      <div className="flow-inputs">
        {modalities.map(({ key, title, detail, tone }) => <div key={key} className="flow-input"><span className={'flow-icon ' + tone}>{key.toUpperCase()}</span><strong>{title}</strong><small>{detail}</small></div>)}
      </div>
      <div className="flow-connection" aria-hidden="true"><i /><i /><i /><i /></div>
      <div className="flow-core"><span aria-hidden="true">◇</span><div><strong>Masked attention fusion</strong><small>Prototype behavior; four-input survival results are pending</small></div></div>
      <div className="flow-output"><span>Relative risk · unvalidated</span><span>Evidence score · not confidence</span></div>
    </div>
  )
}

function Home() {
  return (
    <main id="main" tabIndex={-1}>
      <section className="shell landing">
        <div className="hero-copy">
          <p className="eyebrow">MULTIMODAL ONCOLOGY RESEARCH</p>
          <h1>CancerFusion AI research platform.</h1>
          <p className="hero-summary">Explore a four-branch research prototype and compare it with the survival studies that have actually been evaluated. Current verified performance evidence covers clinical and RNA inputs; image fusion is still a feasibility study.</p>
          <div className="actions">
            <a className="button primary" href="#workspace">Open research workspace <span aria-hidden="true">→</span></a>
            <a className="button secondary" href="#research">Review evaluated evidence</a>
          </div>
        </div>
        <FlowDiagram />
      </section>
      <section className="evidence-band" aria-label="Verified research evidence">
        <div className="shell evidence-grid">
          <div className="evidence-intro"><h2>Evaluated survival evidence</h2><p>Retrospective results; not a diagnostic performance claim.</p></div>
          <div><strong>217 / 66</strong><span>KIRC patients / recorded deaths · clinical + RNA folds</span></div>
          <div><strong>0.7077</strong><span>Best KIRC mean-fold C-index · concatenation</span></div>
          <div><strong>22 / 7</strong><span>Four-input technical cases / deaths · no model fit</span></div>
        </div>
      </section>
      <section className="shell workflow">
        <div className="section-heading"><h2>Work through a case</h2><p>The workspace records both the inputs supplied and the inputs used for each model run.</p></div>
        <div className="workflow-grid">
          <article><h3>Register research inputs</h3><p>Start with synthetic feature vectors or upload arrays that match the Phase 2 checkpoint contract.</p></article>
          <article><h3>Inspect prototype behavior</h3><p>Compare the checkpoint output with an input masked for the same case. This is an engineering demonstration, not a clinical comparison.</p></article>
          <article><h3>Keep results traceable</h3><p>Review the recorded inputs, checkpoint identity and raw scores, with their research-only interpretation.</p></article>
        </div>
      </section>
    </main>
  )
}

const researchSteps = [
  ['Audit the source inputs', 'Check recorded event codes, missing follow-up, per-case modality coverage, and whether radiology files decode through the intended branch.'],
  ['Freeze the evaluation cohort', 'Use distinct patient IDs for training, validation, and held-out testing. Record usable case and event counts for every split. Fit preprocessing on training data.'],
  ['Compare like for like', 'Evaluate a clinical baseline, unimodal models, and multimodal fusion on the same eligible held-out patients. Compare full-input and missing-modality conditions on paired cases.'],
  ['Report measured performance', 'Use a survival concordance metric with uncertainty intervals. Report time-specific discrimination or calibration only when observed outcomes, follow-up, and event counts support them.'],
]

const measuredResults = [
  ['KIRC · age-only Cox', '0.6494', '0.5784–0.7204', 'Reference model'],
  ['KIRC · RNA-only', '0.6865', '0.6091–0.7566', 'Paired gain over Cox not established'],
  ['KIRC · age + RNA concatenation', '0.7077', '0.6315–0.7804', 'Highest point estimate; paired interval includes no gain'],
  ['KIRC · age + RNA attention', '0.7029', '0.6260–0.7683', 'Below concatenation by point estimate'],
  ['BRCA · age + RNA concatenation', '0.6423', '0.5298–0.7489', 'Held-out; paired gain over Cox not established'],
]

function Research() {
  return (
    <main className="shell research-page" id="main" tabIndex={-1}>
      <div className="page-heading"><p className="eyebrow">METHODS & EVIDENCE</p><h1>Evaluated survival research</h1><p>The four-input Phase 2 checkpoint is a prototype. The results below come from separate clinical/RNA studies and do not validate image fusion, diagnosis, treatment selection or patient benefit.</p></div>
      <div className="research-grid">
        <section className="panel research-section">
          <h2>Prototype architecture</h2>
          <p>The Phase 2 checkpoint accepts pathology, radiology, omics and clinical feature arrays and returns a relative risk score, an evidence score, and input-pooling weights. Its outputs demonstrate software behavior on supplied features; this checkpoint has no independently validated four-input survival result.</p>
          <FlowDiagram />
        </section>
        <section className="panel research-section">
          <h2>What the evidence means</h2>
          <p className="research-answer">Survival ranking was measured on clinical/RNA data only.</p>
          <p>The original 122-case pilot used model-derived evaluation labels and had no validation deaths. Its perfect classification scores and 0.78125 C-index are invalid as observed-outcome performance. They are withdrawn from the evidence claims.</p>
          <p>Ensemble spread has not been shown to rank errors reliably and is not calibrated confidence. Four-input imaging data currently form an audited feasibility cohort only.</p>
          <a className="text-link" href="/api/research/handoff">Download research handoff ↗</a>
        </section>
      </div>
      <section className="training-section">
        <div className="section-heading"><h2>Measured C-index results</h2><p>Harrell C summarizes ranking among comparable survival pairs. These are not accuracy, precision, recall or cancer-detection rates. Intervals condition on fitted models and do not include model-refitting uncertainty.</p></div>
        <div className="table-scroll"><table className="results-table"><thead><tr><th>Model / cohort</th><th>Mean C-index</th><th>95% interval</th><th>Interpretation</th></tr></thead><tbody>{measuredResults.map(([model, score, interval, note]) => <tr key={model}><th scope="row">{model}</th><td>{score}</td><td>{interval}</td><td>{note}</td></tr>)}</tbody></table></div>
        <p className="evidence-note">KIRC: five locked folds, 217 patients and 66 deaths. BRCA: 216 held-out patients and 30 deaths. KIRC concatenation versus age Cox: ΔC +0.0582 (95% interval −0.0316 to +0.1478). No local clinical/RNA comparison establishes a multimodal gain. The BRCA row is a different cohort and split.</p>
      </section>
      <section className="training-section">
        <div className="section-heading"><h2>Research status</h2><p>Current limits and next experiments are recorded here; planned work is not presented as completed evidence.</p></div>
        <div className="training-grid">{researchSteps.map(([title, body]) => <article key={title}><h3>{title}</h3><p>{body}</p></article>)}</div>
      </section>
    </main>
  )
}

function FigureGallery({ figures }) {
  const [selected, setSelected] = useState(figures[0]?.id)
  const figure = figures.find(item => item.id === selected)
  if (!figure) return <p className="panel quiet">No figure files are available in the local run bundle.</p>
  return (
    <section className="panel figure-panel">
      <label className="field compact">Figure
        <select value={selected} onChange={event => setSelected(event.target.value)}>
          {figures.map(item => <option key={item.id} value={item.id}>{item.title}</option>)}
        </select>
      </label>
      <figure><img src={figure.url} alt={figure.title} /><figcaption>{figure.caption}</figcaption></figure>
      <a className="button secondary" href={figure.url} download>Download figure</a>
    </section>
  )
}

function Figures() {
  const resource = useResource(loadFigures, [])
  return (
    <main className="shell research-page" id="main" tabIndex={-1}>
      <div className="page-heading"><h1>Research figures</h1><p>Figures are linked to their source reports. Archived pilot classification plots use generated labels and are not valid performance evidence.</p></div>
      {resource.data ? <FigureGallery figures={resource.data} /> : null}
      {resource.error ? <div className="panel quiet" role="alert"><p>{resource.error.message}</p><button className="button secondary" onClick={resource.retry}>Retry connection</button></div> : null}
      {!resource.data && !resource.error ? <p className="panel quiet" role="status">Loading figures…</p> : null}
    </main>
  )
}

function App() {
  const [page, setPage] = useState(currentPage)
  useEffect(() => {
    function navigate() {
      setPage(currentPage())
      window.scrollTo(0, 0)
    }
    window.addEventListener('hashchange', navigate)
    return () => window.removeEventListener('hashchange', navigate)
  }, [])
  useEffect(() => {
    document.title = pageTitles[page] + ' · CancerFusion AI'
  }, [page])

  const content = {
    home: <Home />,
    research: <Research />,
    figures: <Figures />,
    workspace: <Suspense fallback={<main className="shell" id="main" tabIndex={-1}><p role="status">Loading workspace…</p></main>}><Workspace /></Suspense>,
  }[page]
  return (
    <>
      <a className="skip-link" href="#main" onClick={event => { event.preventDefault(); document.getElementById('main')?.focus() }}>Skip to content</a>
      <Header page={page} />
      {content}
      <footer className="footer"><div className="shell"><span>CancerFusion AI · Precision oncology research</span><a href="#research">Study evidence and limitations</a></div></footer>
    </>
  )
}

createRoot(document.getElementById('root')).render(<App />)
