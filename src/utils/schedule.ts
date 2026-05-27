import type { Course } from '../types/course'

function overlaps(
  a: Course['slots'][number],
  b: Course['slots'][number],
) {
  return a.day === b.day && a.startHour < b.endHour && b.startHour < a.endHour
}

export function coursesConflict(a: Course, b: Course) {
  if (a.id === b.id) return false
  return a.slots.some((aSlot) => b.slots.some((bSlot) => overlaps(aSlot, bSlot)))
}

export function getConflictingCourseIds(
  candidate: Course | null | undefined,
  selectedCourses: Course[],
) {
  if (!candidate) return new Set<string>()
  return new Set(
    selectedCourses
      .filter((selected) => coursesConflict(candidate, selected))
      .map((selected) => selected.id),
  )
}
