import { defineConfig, devices } from '@playwright/test'

/** Records the read responses of the real API over the demo dataset (data/demo) into
 *  public/demo/api, for the static online demo. Run after `python -m fda_api.demo_seed`. */
const PORT = 5186
const API_PORT = 8786

export default defineConfig({
  testDir: './demo-record',
  timeout: 60 * 60_000,
  workers: 1,
  reporter: 'list',
  use: { baseURL: `http://127.0.0.1:${PORT}` },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 900 } },
    },
  ],
  webServer: [
    {
      command: `uv run uvicorn fda_api.main:app --host 127.0.0.1 --port ${API_PORT}`,
      url: `http://127.0.0.1:${API_PORT}/v1/health`,
      cwd: '../..',
      env: {
        DATABASE_URL: 'sqlite:///data/demo/demo.db',
        DATA_DIR: 'data/demo',
        FDA_JOB_RUNNER: 'inline',
        FDA_PROVIDERS: 'fake',
        REDIS_URL: '',
      },
      reuseExistingServer: false,
      timeout: 180_000,
    },
    {
      command: `npx vite --host 127.0.0.1 --port ${PORT} --strictPort`,
      url: `http://127.0.0.1:${PORT}`,
      env: { FDA_API_URL: `http://127.0.0.1:${API_PORT}` },
      reuseExistingServer: false,
      timeout: 120_000,
    },
  ],
})
