import { defineConfig, devices } from '@playwright/test'

const PORT = 5175
// overridable when another local project already listens on 8766
const API_PORT = Number(process.env.E2E_API_PORT ?? 8766)

export default defineConfig({
  // Asset pages pull large chart bundles through the dev server: cap parallelism and give
  // first renders time on a cold cache.
  workers: 3,
  expect: { timeout: 30_000 },
  timeout: 90_000,
  testDir: './e2e',
  fullyParallel: true,
  retries: process.env.CI ? 1 : 0,
  reporter: 'list',
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: 'retain-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 900 } },
    },
  ],
  webServer: [
    {
      command: `node -e "require('fs').mkdirSync('data/e2e', { recursive: true })" && uv run alembic upgrade head && uv run uvicorn fda_api.main:app --host 127.0.0.1 --port ${API_PORT}`,
      url: `http://127.0.0.1:${API_PORT}/v1/health`,
      cwd: '../..',
      env: {
        DATABASE_URL: 'sqlite:///data/e2e/e2e.db',
        DATA_DIR: 'data/e2e',
        FDA_JOB_RUNNER: 'inline',
        FDA_PROVIDERS: 'fake',
        // In-memory cache: e2e must not depend on (or pollute) the dev Redis.
        REDIS_URL: '',
        // e2e never calls a paid LLM, whatever keys the machine has
        ...Object.fromEntries(
          [
            'ANTHROPIC',
            'GEMINI',
            'OPENAI',
            'MISTRAL',
            'DEEPSEEK',
            'GROQ',
            'XAI',
            'OPENROUTER',
            'CUSTOM',
          ].map((k) => [`${k}_API_KEY`, '']),
        ),
      },
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
    },
    {
      command: `npx vite --host 127.0.0.1 --port ${PORT} --strictPort`,
      url: `http://127.0.0.1:${PORT}`,
      env: { FDA_API_URL: `http://127.0.0.1:${API_PORT}` },
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
    },
  ],
})
