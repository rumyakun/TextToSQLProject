import type { AuthService } from '../../types/auth'
import { localAuthService } from './localAuthService'
import { remoteAuthService } from './remoteAuthService'

const authMode = import.meta.env.VITE_AUTH_MODE?.trim().toLowerCase()

export const authService: AuthService =
  authMode === 'local' ? localAuthService : remoteAuthService
