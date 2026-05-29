import type { Course, Weekday } from '../types/course'

const dayAliases: Record<string, Weekday | null> = {
  mon: 'Mon',
  monday: 'Mon',
  월: 'Mon',
  월요일: 'Mon',
  tue: 'Tue',
  tues: 'Tue',
  tuesday: 'Tue',
  화: 'Tue',
  화요일: 'Tue',
  wed: 'Wed',
  wednesday: 'Wed',
  수: 'Wed',
  수요일: 'Wed',
  thu: 'Thu',
  thur: 'Thu',
  thurs: 'Thu',
  thursday: 'Thu',
  목: 'Thu',
  목요일: 'Thu',
  fri: 'Fri',
  friday: 'Fri',
  금: 'Fri',
  금요일: 'Fri',
  sat: 'Sat',
  saturday: 'Sat',
  토: 'Sat',
  토요일: 'Sat',
  sun: null,
  sunday: null,
  일: null,
  일요일: null,
}

const dayPattern =
  /(monday|mon|tuesday|tues|tue|wednesday|wed|thursday|thurs|thur|thu|friday|fri|saturday|sat|sunday|sun|월요일|화요일|수요일|목요일|금요일|토요일|일요일|월|화|수|목|금|토|일)/gi

export function normalizeCourseDay(value: string): Weekday | null {
  return dayAliases[value.trim().toLowerCase()] ?? null
}

export function parseCourseClockHour(value: string) {
  const match = value.match(/([0-2]?\d)(?::([0-5]\d))?/)
  if (!match) return null
  const hour = Number(match[1])
  const minute = Number(match[2] ?? '0')
  if (!Number.isFinite(hour) || !Number.isFinite(minute)) return null
  return hour + minute / 60
}

function slotKey(slot: Course['slots'][number]) {
  return `${slot.day}-${slot.startHour}-${slot.endHour}`
}

export function parseCourseSlotsFromText(text: string): Course['slots'] {
  const slots: Course['slots'] = []
  const seen = new Set<string>()
  const ignored = new Set(['', 'time tba', '시간표 미지정', '미지정', 'null'])

  if (ignored.has(text.trim().toLowerCase())) return slots

  for (const part of text.split(/[,;/\n]+/)) {
    const days = [...part.matchAll(dayPattern)]
      .map((match) => normalizeCourseDay(match[0]))
      .filter((day): day is Weekday => day !== null)

    const timeMatch = part.match(
      /([0-2]?\d(?::[0-5]\d)?)\s*(?:-|~|–|—|to)\s*([0-2]?\d(?::[0-5]\d)?)/i,
    )
    if (days.length === 0 || !timeMatch) continue

    const startHour = parseCourseClockHour(timeMatch[1])
    const endHour = parseCourseClockHour(timeMatch[2])
    if (startHour === null || endHour === null || endHour <= startHour) continue

    for (const day of days) {
      const slot = { day, startHour, endHour }
      const key = slotKey(slot)
      if (seen.has(key)) continue
      seen.add(key)
      slots.push(slot)
    }
  }

  return slots
}
