/** Static online demo (Vercel): no server, the engine runs in the browser over exported data. */
export const DEMO = import.meta.env.VITE_DEMO === '1'
export const DEMO_REPO =
  import.meta.env.VITE_DEMO_REPO ?? 'https://github.com/CyrilKFOURI/holdout'
export const DEMO_AS_OF = import.meta.env.VITE_DEMO_AS_OF ?? ''
export const LOCKED_TYPE = '/errors/demo-locked'
export const LOCKED_MESSAGE =
  'Full version only: this action needs the server (data ingestion, saving, orders, AI agent). Run the full app from GitHub to use it.'
