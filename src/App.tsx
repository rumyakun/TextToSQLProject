import { useEffect, useState } from 'react'
import AuthPage from './pages/AuthPage'
import MainPage from './pages/MainPage'
import { authService } from './services/auth'
import type { AuthUser } from './types/auth'

export default function App() {
  const [currentUser, setCurrentUser] = useState<AuthUser | null>(null)
  const [authPageOpen, setAuthPageOpen] = useState(false)

  useEffect(() => {
    let disposed = false

    async function loadCurrentUser() {
      const user = await authService.getCurrentUser()
      if (!disposed) {
        setCurrentUser(user)
      }
    }

    void loadCurrentUser()

    return () => {
      disposed = true
    }
  }, [])

  async function handleLogin(studentNo: string) {
    try {
      const user = await authService.login(studentNo)
      setCurrentUser(user)
      setAuthPageOpen(false)
      return null
    } catch (error) {
      if (error instanceof Error) {
        return error.message
      }
      return 'Login failed.'
    }
  }

  async function handleLogout() {
    await authService.logout()
    setCurrentUser(null)
  }

  if (authPageOpen) {
    return (
      <AuthPage
        onLogin={handleLogin}
        onBack={() => setAuthPageOpen(false)}
      />
    )
  }

  return (
    <MainPage
      userStudentNo={currentUser?.studentNo}
      userName={currentUser?.name}
      completedCourses={currentUser?.completedCourses}
      onLoginClick={() => setAuthPageOpen(true)}
      onLogout={handleLogout}
    />
  )
}
