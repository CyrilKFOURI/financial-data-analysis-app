import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    // The tree-shaken ECharts vendor chunk is ~575 kB (190 kB gzip), loaded only on asset pages.
    chunkSizeWarningLimit: 600,
    rolldownOptions: {
      output: {
        // Chart engines change rarely and are heavy: cache them apart from app code.
        advancedChunks: {
          groups: [
            { name: 'echarts', test: /node_modules[\\/](echarts|zrender)/ },
            { name: 'lightweight-charts', test: /node_modules[\\/]lightweight-charts/ },
            { name: 'dockview', test: /node_modules[\\/]dockview/ },
          ],
        },
      },
    },
  },
  // Pre-bundle chart engines: discovering them at runtime triggers a reload mid-test.
  optimizeDeps: {
    include: [
      'echarts/core',
      'echarts/charts',
      'echarts/components',
      'echarts/renderers',
      'lightweight-charts',
    ],
  },
  server: {
    proxy: { '/v1': process.env.FDA_API_URL ?? 'http://127.0.0.1:8000' },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
  },
})
