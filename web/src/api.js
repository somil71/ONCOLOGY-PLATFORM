import { useEffect, useState } from 'react'

export const modalities = [
  { key: 'wsi', title: 'Pathology', detail: 'Tissue morphology', tone: 'lavender' },
  { key: 'rad', title: 'Radiology', detail: 'Imaging features', tone: 'blue' },
  { key: 'rna', title: 'Omics', detail: 'Molecular features', tone: 'green' },
  { key: 'cli', title: 'Clinical', detail: 'Patient variables', tone: 'peach' },
]

export async function request(path, { data, signal } = {}) {
  const response = await fetch('/api' + path, {
    signal,
    ...(data === undefined ? {} : {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    }),
  })
  const body = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = body?.detail
    const message = typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map(item => item.msg).join(' ') : ''
    throw new Error(message || 'The Python backend could not complete this request.')
  }
  if (body === null) throw new Error('The Python backend returned an unexpected response.')
  return body
}

export function useResource(load, dependencies) {
  const [result, setResult] = useState({ data: null, error: null })
  const [revision, setRevision] = useState(0)
  useEffect(() => {
    const controller = new AbortController()
    setResult({ data: null, error: null })
    load(controller.signal).then(
      data => { if (!controller.signal.aborted) setResult({ data, error: null }) },
      error => { if (!controller.signal.aborted) setResult({ data: null, error }) },
    )
    return () => controller.abort()
  }, [revision, ...dependencies])
  return { ...result, retry: () => setRevision(current => current + 1) }
}
