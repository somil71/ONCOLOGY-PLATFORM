import { defineConfig } from 'vite'

const api = { '/api': 'http://127.0.0.1:8000' }

export default defineConfig({
  esbuild: { jsx: 'automatic' },
  server: { host: '127.0.0.1', port: 5173, proxy: api },
  preview: { host: '127.0.0.1', port: 4173, proxy: api },
})
