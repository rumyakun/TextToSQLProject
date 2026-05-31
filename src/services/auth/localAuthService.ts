import type { AuthService, AuthUser } from '../../types/auth'
import { sha256 } from '../../utils/crypto'

type StoredUser = AuthUser & {
  passwordHash: string
}

const USERS_STORAGE_KEY = 'course-frontend-users'
const SESSION_STORAGE_KEY = 'course-frontend-current-user-student-no'

function normalizeStudentNo(studentNo: string) {
  return studentNo.trim()
}

function readUsers(): StoredUser[] {
  const raw = localStorage.getItem(USERS_STORAGE_KEY)
  if (!raw) return []

  try {
    const parsed = JSON.parse(raw) as unknown
    if (!Array.isArray(parsed)) return []
    return parsed.filter(
      (item): item is StoredUser =>
        !!item &&
        typeof item === 'object' &&
        typeof item.name === 'string' &&
        typeof item.studentNo === 'string' &&
        typeof item.passwordHash === 'string',
    )
  } catch {
    return []
  }
}

function writeUsers(users: StoredUser[]) {
  localStorage.setItem(USERS_STORAGE_KEY, JSON.stringify(users))
}

function sanitizeUser(user: StoredUser): AuthUser {
  return {
    id: user.id,
    name: user.name,
    studentNo: user.studentNo,
    departmentCode: user.departmentCode,
    departmentName: user.departmentName,
    completedCourses: user.completedCourses,
  }
}

export const localAuthService: AuthService = {
  async getCurrentUser() {
    const studentNo = localStorage.getItem(SESSION_STORAGE_KEY)
    if (!studentNo) return null
    const users = readUsers()
    const found = users.find((user) => user.studentNo === studentNo)
    return found ? sanitizeUser(found) : null
  },

  async login(studentNo) {
    const normalizedStudentNo = normalizeStudentNo(studentNo)
    const users = readUsers()
    const found = users.find((user) => user.studentNo === normalizedStudentNo)
    if (!found) {
      throw new Error('등록되지 않은 학번입니다.')
    }

    localStorage.setItem(SESSION_STORAGE_KEY, normalizedStudentNo)
    return sanitizeUser(found)
  },

  async signup(name, studentNo, password, departmentCode) {
    const normalizedStudentNo = normalizeStudentNo(studentNo)
    const users = readUsers()
    if (users.some((user) => user.studentNo === normalizedStudentNo)) {
      throw new Error('이미 등록된 학번입니다.')
    }

    const hash = await sha256(password)
    const nextUser: StoredUser = {
      id: crypto.randomUUID(),
      name: name.trim(),
      studentNo: normalizedStudentNo,
      departmentCode,
      passwordHash: hash,
    }
    writeUsers([...users, nextUser])
    localStorage.setItem(SESSION_STORAGE_KEY, normalizedStudentNo)
    return sanitizeUser(nextUser)
  },

  async logout() {
    localStorage.removeItem(SESSION_STORAGE_KEY)
  },
}
