import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { RouterProvider } from '@tanstack/react-router'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import { DEMO } from './demo/config'
import { router } from './router'

const queryClient = new QueryClient()

// online demo: start loading the in-browser engine while the first page renders
if (DEMO) void import('./demo/server').then((m) => m.warmEngine())

const root = document.getElementById('root')
if (!root) throw new Error('Missing #root element')

createRoot(root).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </StrictMode>,
)
