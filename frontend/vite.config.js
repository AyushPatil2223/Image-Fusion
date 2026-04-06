import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Tailwind v3 works via PostCSS (postcss.config.js) — no vite plugin needed
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
    },
  },
})
