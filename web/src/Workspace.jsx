import { useState } from 'react'
import { modalities, request, useResource } from './api'

const dateLabel = new Intl.DateTimeFormat('en', { dateStyle: 'medium', timeStyle: 'short' })
const percent = new Intl.NumberFormat('en', { style: 'percent', maximumFractionDigits: 1 })
const decimal = new Intl.NumberFormat('en', { maximumFractionDigits: 5 })
const loadWorkspace = signal => Promise.all([request('/cases', { signal }), request('/model', { signal })])

function InputForm({ model, onCreated, onClose }) {
  const [code, setCode] = useState('')
  const [source, setSource] = useState('demo')
  const [file, setFile] = useState(null)
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)

  async function save(event) {
    event.preventDefault()
    setSaving(true)
    setError('')
    try {
      const data = { code, source }
      if (source === 'upload') {
        if (!file) throw new Error('Choose a JSON feature file.')
        if (file.size > 2_000_000) throw new Error('Use a feature file under 2 MB.')
        try {
          data.features = JSON.parse(await file.text())
        } catch {
          throw new Error('Choose a file that contains valid JSON.')
        }
      }
      onCreated(await request('/cases', { data }))
    } catch (failure) {
      setError(failure.message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <form className="panel create-panel" onSubmit={save}>
      <div className="panel-head"><div><h2>Create a case</h2><p>Use a de-identified study code such as BRCA-001.</p></div></div>
      <div className="form-fields">
        <label className="field">Case code
          <input value={code} onChange={event => setCode(event.target.value)} placeholder="BRCA-001" required minLength={3} maxLength={40} />
        </label>
        <fieldset className="source-options">
          <legend>Input source</legend>
          <label><input type="radio" name="source" checked={source === 'demo'} onChange={() => setSource('demo')} /> Synthetic demonstration</label>
          <label><input type="radio" name="source" checked={source === 'upload'} onChange={() => setSource('upload')} /> Upload features</label>
        </fieldset>
        {source === 'upload' ? (
          <div className="upload-block">
            <label className="field">Preprocessed feature JSON
              <input type="file" accept=".json,application/json" onChange={event => setFile(event.target.files[0] || null)} required />
            </label>
            <p>Supply one or more modality arrays in the same feature order used for training.</p>
            <div className="shape-list">
              {modalities.map(({ key }) => <code key={key}>{key} [{model.feature_shapes[key]?.join(' × ')}]</code>)}
            </div>
          </div>
        ) : <p className="field-note">This case uses generated feature vectors for a reproducible demonstration.</p>}
        {error ? <p role="alert" className="error-note">{error}</p> : null}
        <div className="actions">
          <button className="button primary" disabled={saving || !model.available}>{saving ? 'Saving…' : 'Save case'}</button>
          <button type="button" className="button secondary" onClick={onClose} disabled={saving}>Cancel</button>
        </div>
      </div>
    </form>
  )
}

function RunReport({ caseId, runs }) {
  const [selectedId, setSelectedId] = useState('')
  const run = runs.find(item => item.run_id === selectedId) || runs[0]
  if (!run) return <section className="panel quiet"><h2>Result history</h2><p>Run a comparison to see the model scores and modality weights here.</p></section>
  const withheld = modalities.filter(({ key }) => run.available_modalities.includes(key) && !run.selected_modalities.includes(key))
  const heldOutNames = withheld.map(({ title }) => title).join(', ')
  const reportPath = '/api/cases/' + caseId + '/runs/' + run.run_id + '/report'

  return (
    <section className="panel result-panel">
      <div className="panel-head">
        <div><h2>Recorded comparison</h2><p>{run.input_source} · {dateLabel.format(new Date(run.created_at))}</p></div>
        <label className="field compact">History
          <select value={run.run_id} onChange={event => setSelectedId(event.target.value)}>
            {runs.map(item => <option key={item.run_id} value={item.run_id}>
              {dateLabel.format(new Date(item.created_at))} · {item.selected_modalities.join(', ').toUpperCase()}
            </option>)}
          </select>
        </label>
      </div>
      <div className="result-reading">
        <strong>{withheld.length ? `Withheld in this run: ${heldOutNames}` : 'Both passes used the same inputs'}</strong>
        <p>{withheld.length ? 'The two scores show how this checkpoint responds when those inputs are masked for the same case.' : 'Withhold an available input and rerun to see how it affects the model output.'}</p>
      </div>
      <div className="scores">
        {run.outputs.map(output => (
          <div key={output.condition}>
            <h3>{output.condition}</h3>
            <dl><div><dt>Relative risk</dt><dd>{decimal.format(output.risk)}</dd></div><div><dt>Uncertainty</dt><dd>{decimal.format(output.uncertainty)}</dd></div></dl>
          </div>
        ))}
      </div>
      <dl className="score-guide">
        <div><dt>Relative risk</dt><dd>Higher scores rank cases as higher risk. The score has no calibrated survival probability or time horizon.</dd></div>
        <div><dt>Evidence score</dt><dd>A raw model output. It is not a probability of error or calibrated confidence, and it has no validated clinical threshold.</dd></div>
      </dl>
      <div className="weights">
        <div className="weights-heading"><h3>Modality pooling weights</h3><div><span className="legend baseline" /> Baseline <span className="legend selected" /> Selected</div></div>
        {modalities.map(({ key, title }) => (
          <div className="weights-row" key={key}>
            <span>{title}</span>
            <div className="weight-pair">
              {run.outputs.map((output, index) => (
                <div className="bar-row" key={output.condition}>
                  <div className="track"><div className={index === 0 ? 'baseline' : 'selected'} style={{ width: percent.format(output.weights[key]) }} /></div>
                  <output>{percent.format(output.weights[key])}</output>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="report-footer"><p>Pooling weights describe this checkpoint's internal allocation only; they do not establish modality value or clinical importance. The current independently evaluated survival studies use clinical and RNA inputs, not this four-input checkpoint.</p><div className="actions">
        <a className="button secondary" href={reportPath + '?format=markdown'}>Download report</a>
        <a className="button secondary" href={reportPath + '?format=json'}>JSON</a>
      </div></div>
    </section>
  )
}

function CaseDetails({ caseId, model, onRunSaved }) {
  const detail = useResource(signal => request('/cases/' + caseId, { signal }), [caseId])
  if (detail.error) return <div className="panel quiet" role="alert">{detail.error.message}<button className="button secondary" onClick={detail.retry}>Retry</button></div>
  if (!detail.data) return <p className="panel quiet" role="status">Loading the selected case…</p>
  return <CaseComparison details={detail.data} model={model} onRunSaved={onRunSaved} />
}

function CaseComparison({ details, model, onRunSaved }) {
  const [selection, setSelection] = useState(details.available_modalities)
  const [runs, setRuns] = useState(details.runs)
  const [reportVersion, setReportVersion] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const withheld = modalities.filter(({ key }) => details.available_modalities.includes(key) && !selection.includes(key))
  const isDemo = details.source === 'synthetic demonstration'

  function toggleInput(key, checked) {
    setSelection(current => checked ? [...current, key] : current.filter(item => item !== key))
  }

  async function runComparison() {
    setBusy(true)
    setError('')
    try {
      const run = await request('/cases/' + details.id + '/runs', { data: { selected_modalities: selection } })
      const next = [run, ...runs.filter(item => item.run_id !== run.run_id)]
      setRuns(next)
      setReportVersion(current => current + 1)
      onRunSaved(details.id, next.length)
    } catch (failure) {
      setError(failure.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <section className="panel case-panel">
        <div className="panel-head"><div><h2>{details.code}</h2><p>{isDemo ? 'Synthetic demonstration' : 'Uploaded research features'} · {details.available_modalities.length} available inputs</p></div></div>
        <div className="panel-content">
          <h3>Comparison inputs</h3>
          <p>The baseline uses every input supplied for this case. Uncheck a modality to withhold it from the second pass. This compares checkpoint behavior on one feature record; it does not measure clinical accuracy.</p>
          {isDemo ? <p className="demo-note">This case uses generated features. Its results illustrate model behavior on an example.</p> : null}
          <fieldset className="modality-select" disabled={busy}>
            <legend className="sr-only">Comparison modalities</legend>
            {modalities.map(({ key, title, detail, tone }) => {
              const available = details.available_modalities.includes(key)
              return (
                <label className={'modality-choice ' + tone} key={key}>
                  <input type="checkbox" checked={selection.includes(key)} disabled={!available}
                    onChange={event => toggleInput(key, event.target.checked)} />
                  <span><strong>{title}</strong><small>{available ? `${detail} · ${selection.includes(key) ? 'Included' : 'Withheld'}` : 'Unavailable for this case'}</small></span>
                </label>
              )
            })}
          </fieldset>
          {error ? <p role="alert" className="error-note">{error}</p> : null}
          <div className="comparison-action">
            <p role="status">{selection.length === 0 ? 'Include at least one modality.' : withheld.length ? `Withholding ${withheld.map(({ title }) => title).join(', ')}` : 'All supplied inputs are selected.'}</p>
            <button className="button primary" disabled={busy || !model.available || selection.length === 0} onClick={runComparison}>
              {busy ? 'Running the checkpoint…' : 'Run and record comparison'}
            </button>
          </div>
        </div>
      </section>
      <RunReport key={reportVersion} caseId={details.id} runs={runs} />
    </>
  )
}

function CaseWorkspace({ initialCases, model }) {
  const [cases, setCases] = useState(initialCases)
  const [selectedId, setSelectedId] = useState(initialCases[0]?.id ?? '')
  const [showForm, setShowForm] = useState(false)
  const [query, setQuery] = useState('')
  const visible = cases.filter(item => item.code.toLowerCase().includes(query.trim().toLowerCase()))

  function addCase(detail) {
    setCases(current => [{ ...detail, run_count: detail.runs.length }, ...current])
    setSelectedId(detail.id)
    setShowForm(false)
    setQuery('')
  }

  function updateCount(id, count) {
    setCases(current => current.map(item => item.id === id ? { ...item, run_count: count } : item))
  }

  return (
    <>
      <div className="workspace-meta">
        <span className={'status ' + (model.available ? 'ready' : 'pending')}>{model.available ? 'Checkpoint available' : 'Checkpoint unavailable'}</span>
        {model.checkpoint ? <span className="file-label">{model.checkpoint.name}</span> : null}
        <span className="count-label">{cases.length} {cases.length === 1 ? 'saved case' : 'saved cases'}</span>
      </div>
      <div className="workspace-grid">
        <aside className="panel library">
          <div className="panel-head"><h2>Case library</h2><button className="button secondary small" onClick={() => setShowForm(true)} disabled={!model.available}>New case</button></div>
          <label className="sr-only" htmlFor="search">Search case codes</label>
          <input id="search" type="search" value={query} onChange={event => setQuery(event.target.value)} placeholder="Search case codes" />
          {visible.length > 0 ? (
            <ul className="case-list">
              {visible.map(item => <li key={item.id}>
                <button aria-current={selectedId === item.id ? 'true' : undefined} onClick={() => { setSelectedId(item.id); setShowForm(false) }}>
                  <span><strong>{item.code}</strong><small>{item.source}</small></span>
                  <span className="case-count">{item.run_count} {item.run_count === 1 ? 'run' : 'runs'}</span>
                </button>
              </li>)}
            </ul>
          ) : <p className="library-empty">{cases.length ? 'No matching cases.' : 'No cases saved yet.'}</p>}
        </aside>
        <div className="case-content">
          {showForm ? <InputForm model={model} onCreated={addCase} onClose={() => setShowForm(false)} /> : null}
          {selectedId ? <CaseDetails key={selectedId} caseId={selectedId} model={model} onRunSaved={updateCount} /> : null}
          {!selectedId && !showForm ? <section className="panel quiet"><h2>Start with a case</h2><p>Create a synthetic demonstration or upload compatible, de-identified feature arrays.</p><button className="button primary" onClick={() => setShowForm(true)} disabled={!model.available}>Create case</button></section> : null}
        </div>
      </div>
    </>
  )
}

export default function Workspace() {
  const resource = useResource(loadWorkspace, [])
  return (
    <main className="shell workspace-page" id="main" tabIndex={-1}>
      <header className="workspace-intro">
        <div><h1>Case workspace</h1><p>See how the same case scores with and without an input.</p></div>
        <aside aria-label="How to run a comparison">
          <strong>Research prototype only</strong>
          <p>Four-input outputs are not validated survival predictions. Synthetic examples illustrate checkpoint behavior; do not use scores for diagnosis or care.</p>
          <ol>
            <li>Open a saved case or create a demonstration.</li>
            <li>Uncheck Radiology or another available input.</li>
            <li>Run both conditions and review the saved result.</li>
          </ol>
        </aside>
      </header>
      {resource.error ? <div className="panel quiet" role="alert"><h2>Workspace unavailable</h2><p>{resource.error.message}</p><button className="button secondary" onClick={resource.retry}>Retry connection</button></div> : null}
      {resource.data ? <CaseWorkspace initialCases={resource.data[0]} model={resource.data[1]} /> : null}
      {!resource.data && !resource.error ? <p className="panel quiet" role="status">Loading cases and checkpoint status…</p> : null}
    </main>
  )
}
