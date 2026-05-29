import type { Course } from '../types/course'

export function normalizeCourseCode(value: string) {
  return value.trim().toLowerCase()
}

function courseCodeVariants(value: string) {
  const normalized = normalizeCourseCode(value)
  const variants = [normalized]
  const parts = normalized.split('-')
  if (parts.length >= 3) {
    variants.push(parts.slice(0, -1).join('-'))
  }
  return variants
}

function readCourseCode(value: unknown): string | null {
  if (typeof value === 'string' || typeof value === 'number') {
    const text = String(value).trim()
    return text || null
  }

  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  const record = value as Record<string, unknown>
  const keys = ['subject_code', 'subjectCode', 'courseCode', 'course_id', 'courseId', 'id']

  for (const key of keys) {
    const raw = record[key]
    if (typeof raw === 'string' || typeof raw === 'number') {
      const text = String(raw).trim()
      if (text) return text
    }
  }

  return null
}

export function extractCompletedCourseCodes(completedCourses?: unknown[]) {
  if (!completedCourses) return new Set<string>()

  return new Set(
    completedCourses
      .map(readCourseCode)
      .filter((code): code is string => code !== null)
      .flatMap(courseCodeVariants),
  )
}

export function getUnmetPrerequisiteCodes(
  course: Course,
  completedCourseCodes: Set<string> | null,
) {
  if (!completedCourseCodes) return []

  return (course.prerequisiteCourseCodes ?? []).filter(
    (code) => !completedCourseCodes.has(normalizeCourseCode(code)),
  )
}

export function hasUnmetPrerequisite(
  course: Course,
  completedCourseCodes: Set<string> | null,
) {
  return getUnmetPrerequisiteCodes(course, completedCourseCodes).length > 0
}

export function getUnmetPrerequisiteNames(course: Course, unmetCodes: string[]) {
  const names = course.prerequisiteCourseNames ?? []
  const codes = course.prerequisiteCourseCodes ?? []

  return unmetCodes.map((code) => {
    const index = codes.findIndex(
      (candidate) => normalizeCourseCode(candidate) === normalizeCourseCode(code),
    )
    return index >= 0 ? names[index] : undefined
  })
}

export function formatPrerequisites(codes: string[], names: Array<string | undefined>) {
  return codes
    .map((code, index) => {
      const name = names[index]
      return name ? `${code} ${name}` : code
    })
    .join(', ')
}
