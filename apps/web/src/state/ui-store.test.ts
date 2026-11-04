import { useUiStore } from './ui-store'

beforeEach(() => useUiStore.setState({ theme: 'graphite', drawerMetricId: null }))

test('toggles between graphite and paper', () => {
  useUiStore.getState().toggleTheme()
  expect(useUiStore.getState().theme).toBe('paper')
  useUiStore.getState().toggleTheme()
  expect(useUiStore.getState().theme).toBe('graphite')
})

test('opens and closes the formula drawer', () => {
  useUiStore.getState().openDrawer('R-005')
  expect(useUiStore.getState().drawerMetricId).toBe('R-005')
  useUiStore.getState().closeDrawer()
  expect(useUiStore.getState().drawerMetricId).toBeNull()
})
