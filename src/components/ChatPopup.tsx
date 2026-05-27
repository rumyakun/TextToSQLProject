import { useEffect, useMemo, useRef, useState } from 'react'
import type { Course, CourseDetail } from '../types/course'
import { cn } from '../utils/cn'
import { getConflictingCourseIds } from '../utils/schedule'
import Timetable from './Timetable'

type ChatMessage = {
  id: string
  role: 'user' | 'system'
  text: string
}

type QueryApiResponse = {
  sql?: string
  data?: unknown[]
  warning?: string
  error?: string
}

type QueryRow = Record<string, unknown>
type QueryScheduleSlot = {
  day?: unknown
  start?: unknown
  end?: unknown
  room?: unknown
}

function departmentColor({
  selected,
  hasConflict,
  closed,
}: {
  selected: boolean
  hasConflict: boolean
  closed: boolean
}) {
  if (selected) return 'text-rose-700'
  if (hasConflict) return 'text-amber-700'
  if (closed) return 'text-slate-900'
  return 'text-blue-700'
}

function makeId(prefix: string) {
  return `${prefix}_${Math.random().toString(16).slice(2)}_${Date.now()}`
}

function isQueryRow(value: unknown): value is QueryRow {
  return !!value && typeof value === 'object' && !Array.isArray(value)
}

function readString(row: QueryRow, keys: string[]) {
  for (const key of keys) {
    const value = row[key]
    if (typeof value === 'string' && value.trim()) return value.trim()
    if (typeof value === 'number') return String(value)
  }
  return ''
}

function readNumber(row: QueryRow, keys: string[]) {
  for (const key of keys) {
    const value = row[key]
    if (typeof value === 'number' && Number.isFinite(value)) return value
    if (typeof value === 'string' && value.trim()) {
      const parsed = Number(value)
      if (Number.isFinite(parsed)) return parsed
      const match = value.match(/\d+/)
      if (match) return Number(match[0])
    }
  }
  return undefined
}

function readDetails(row: QueryRow): CourseDetail[] {
  const labels: Array<[string, string]> = [
    ['course_year', '학년'],
    ['subject_code', '과목코드'],
    ['section', '분반'],
    ['subject_name', '과목명'],
    ['category', '이수구분'],
    ['credit_hours', '학점'],
    ['target_year', '대상학년'],
    ['professor', '교수'],
    ['capacity', '정원'],
    ['enrolled', '수강인원'],
    ['grading_method', '성적평가'],
    ['eval_type', '평가방식'],
    ['class_mode', '수업방식'],
    ['dept_name', '학과'],
    ['day_of_week', '요일'],
    ['start_time', '시작'],
    ['end_time', '종료'],
    ['classroom', '강의실'],
  ]

  return labels.flatMap(([key, label]) => {
    const value = row[key]
    if (value === undefined || value === null || value === '') return []
    return [{ label, value: String(value) }]
  })
}

function detailTitle(course: Course) {
  const details = course.details ?? []
  if (details.length === 0) {
    return [
      course.name,
      `교수: ${course.professor}`,
      `시간: ${course.timeText}`,
      course.locationText ? `강의실: ${course.locationText}` : null,
    ]
      .filter(Boolean)
      .join('\n')
  }
  return details.map((detail) => `${detail.label}: ${detail.value}`).join('\n')
}

function normalizeDay(value: string): Course['slots'][number]['day'] | null {
  const normalized = value.trim().toLowerCase()
  if (['mon', 'monday', '월', '월요일'].includes(normalized)) return 'Mon'
  if (['tue', 'tues', 'tuesday', '화', '화요일'].includes(normalized)) return 'Tue'
  if (['wed', 'wednesday', '수', '수요일'].includes(normalized)) return 'Wed'
  if (['thu', 'thur', 'thurs', 'thursday', '목', '목요일'].includes(normalized)) return 'Thu'
  if (['fri', 'friday', '금', '금요일'].includes(normalized)) return 'Fri'
  if (['sat', 'saturday', '토', '토요일'].includes(normalized)) return 'Sat'
  return null
}

function parseClockHour(value: string) {
  const match = value.match(/(\d{1,2})(?::([0-5]\d))?/)
  if (!match) return null
  const hour = Number(match[1])
  const minute = Number(match[2] ?? '0')
  if (!Number.isFinite(hour) || !Number.isFinite(minute)) return null
  return hour + minute / 60
}

function parseSlots(timeText: string): Course['slots'] {
  const slots: Course['slots'] = []
  const parts = timeText.split(/[,;/]+/)

  for (const part of parts) {
    const match = part.match(
      /(mon|monday|tue|tues|tuesday|wed|wednesday|thu|thur|thurs|thursday|fri|friday|sat|saturday|월요일?|화요일?|수요일?|목요일?|금요일?|토요일?)\s*[\s(]*([0-2]?\d(?::[0-5]\d)?)\s*[-~]\s*([0-2]?\d(?::[0-5]\d)?)/i,
    )
    if (!match) continue

    const day = normalizeDay(match[1])
    const startHour = parseClockHour(match[2])
    const endHour = parseClockHour(match[3])
    if (!day || startHour === null || endHour === null || endHour <= startHour) continue
    slots.push({ day, startHour, endHour })
  }

  return slots
}

function readScheduleSlots(row: QueryRow): Course['slots'] {
  const rawSchedule = row.schedule
  if (!Array.isArray(rawSchedule)) return []

  return rawSchedule
    .map((slot): Course['slots'][number] | null => {
      if (!slot || typeof slot !== 'object' || Array.isArray(slot)) return null
      const scheduleSlot = slot as QueryScheduleSlot
      const dayText = readString(scheduleSlot as QueryRow, ['day'])
      const startText = readString(scheduleSlot as QueryRow, ['start'])
      const endText = readString(scheduleSlot as QueryRow, ['end'])
      const day = normalizeDay(dayText)
      const startHour = parseClockHour(startText)
      const endHour = parseClockHour(endText)
      if (!day || startHour === null || endHour === null || endHour <= startHour) {
        return null
      }
      return { day, startHour, endHour }
    })
    .filter((slot): slot is Course['slots'][number] => slot !== null)
}

function readFieldSlot(row: QueryRow): Course['slots'] {
  const dayText = readString(row, ['day_of_week', 'day'])
  const startText = readString(row, ['start_time', 'start'])
  const endText = readString(row, ['end_time', 'end'])
  const day = normalizeDay(dayText)
  const startHour = parseClockHour(startText)
  const endHour = parseClockHour(endText)
  if (!day || startHour === null || endHour === null || endHour <= startHour) {
    return []
  }
  return [{ day, startHour, endHour }]
}

function rowToCourse(row: QueryRow, index: number): Course {
  const subjectCode = readString(row, [
    'subject_code',
    'course_code',
    'course_id',
    'id',
    'code',
  ])
  const section = readString(row, ['section', 'class_no', 'division'])
  const id = subjectCode
    ? section
      ? `${subjectCode}-${section}`
      : subjectCode
    : `DB-${index + 1}`
  const name =
    readString(row, ['subject_name', 'course_name', 'name', 'title']) ||
    `Course ${index + 1}`
  const professor =
    readString(row, ['professor', 'instructor', 'teacher', '교수']) || '-'
  const credits = readNumber(row, ['credit_hours', 'credits', 'credit', '학점']) ?? 0
  const timeText =
    readString(row, ['lecture_time', 'time_text', 'time', '시간']) ||
    'Time TBA'
  const slots = [
    ...readScheduleSlots(row),
    ...readFieldSlot(row),
    ...parseSlots(timeText),
  ]
  const locationText = readString(row, ['classroom', 'room', 'locationText', 'location'])
  const capacity = readNumber(row, ['capacity', '정원'])
  const enrolled = readNumber(row, ['enrolled', 'registered', '수강인원'])
  const status =
    capacity !== undefined && enrolled !== undefined && enrolled >= capacity
      ? 'Closed'
      : 'Open'

  return {
    id,
    name,
    departmentName: readString(row, ['dept_name', 'departmentName', 'department_name']),
    professor,
    credits,
    status,
    capacity,
    enrolled,
    timeText,
    locationText,
    slots,
    details: readDetails(row),
  }
}

function slotKey(slot: Course['slots'][number]) {
  return `${slot.day}-${slot.startHour}-${slot.endHour}`
}

function mergeCourses(existing: Course, next: Course): Course {
  const seenSlots = new Set(existing.slots.map(slotKey))
  const mergedSlots = [...existing.slots]
  for (const slot of next.slots) {
    const key = slotKey(slot)
    if (seenSlots.has(key)) continue
    seenSlots.add(key)
    mergedSlots.push(slot)
  }

  const timeParts = existing.timeText
    .split(/\s*,\s*/)
    .filter(Boolean)
  for (const part of next.timeText.split(/\s*,\s*/).filter(Boolean)) {
    if (part === 'Time TBA' || timeParts.includes(part)) continue
    timeParts.push(part)
  }

  return {
    ...existing,
    timeText: timeParts.length > 0 ? timeParts.join(', ') : existing.timeText,
    locationText: mergeTextList(existing.locationText, next.locationText),
    slots: mergedSlots,
    details: [...(existing.details ?? []), ...(next.details ?? [])],
  }
}

function mergeTextList(existing?: string, next?: string) {
  const parts = (existing ?? '').split(/\s*,\s*/).filter(Boolean)
  for (const part of (next ?? '').split(/\s*,\s*/).filter(Boolean)) {
    if (!parts.includes(part)) parts.push(part)
  }
  return parts.join(', ') || undefined
}

function rowsToCourses(rows: unknown[]) {
  const courses = new Map<string, Course>()
  rows.filter(isQueryRow).forEach((row, index) => {
    const course = rowToCourse(row, index)
    const existing = courses.get(course.id)
    courses.set(course.id, existing ? mergeCourses(existing, course) : course)
  })
  return [...courses.values()]
}

export default function ChatPopup({
  open,
  onClose,
  selectedCourses,
  selectedIds,
  onAddCourse,
  onRemoveCourse,
  onReplaceCourse,
  onOpenExpandedTimetable,
}: {
  open: boolean
  onClose: () => void
  selectedCourses: Course[]
  selectedIds: Set<string>
  onAddCourse: (course: Course) => void
  onRemoveCourse: (courseId: string) => void
  onReplaceCourse: (course: Course) => void
  onOpenExpandedTimetable: () => void
}) {
  const [mounted, setMounted] = useState(false)
  const [visible, setVisible] = useState(false)
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: makeId('sys'),
      role: 'system',
      text: '안녕하세요! 원하는 과목 키워드를 입력하면 검색 결과와 시간표를 보여드릴게요.',
    },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [lastSql, setLastSql] = useState('')
  const [lastWarning, setLastWarning] = useState('')
  const [lastError, setLastError] = useState('')
  const [dbCourses, setDbCourses] = useState<Course[] | null>(null)
  const [hoveredCourse, setHoveredCourse] = useState<Course | null>(null)

  const scrollRef = useRef<HTMLDivElement | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    if (!open) {
      setVisible(false)
      const t = window.setTimeout(() => setMounted(false), 180)
      return () => window.clearTimeout(t)
    }
    setMounted(true)
    const t = window.setTimeout(() => setVisible(true), 10)
    return () => window.clearTimeout(t)
  }, [open])

  useEffect(() => {
    if (!mounted) return
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [mounted, onClose])

  useEffect(() => {
    if (!mounted) return
    const el = scrollRef.current
    if (!el) return
    el.scrollTop = el.scrollHeight
  }, [mounted, messages.length])

  const results = useMemo(() => {
    if (dbCourses === null) return []
    return dbCourses
  }, [dbCourses])
  const hoveredConflictIds = useMemo(
    () => getConflictingCourseIds(hoveredCourse, selectedCourses),
    [hoveredCourse, selectedCourses],
  )

  function cancelQuery() {
    abortRef.current?.abort()
    abortRef.current = null
    setLoading(false)
    setLastError('')
    setLastWarning('')
    setDbCourses(null)
    setMessages((prev) => [
      ...prev,
      {
        id: makeId('sys'),
        role: 'system',
        text: '질의를 취소하셨습니다. 다시 질문해 주세요!',
      },
    ])
  }

  async function send() {
    if (loading) {
      cancelQuery()
      return
    }

    const text = input.trim()
    if (!text) return
    setInput('')
    setLastError('')
    setLastWarning('')
    setDbCourses(null)

    setMessages((prev) => [
      ...prev,
      { id: makeId('usr'), role: 'user', text },
    ])

    setLoading(true)
    const controller = new AbortController()
    abortRef.current = controller

    try {
      const res = await fetch('/api/v1/chat/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: text }),
        signal: controller.signal,
      })
      const payload = (await res.json()) as QueryApiResponse

      if (!res.ok || payload.error) {
        const errorText =
          payload.error ??
          `요청 처리에 실패했습니다. (${res.status})`
        setLastError(errorText)
        setDbCourses(null)
        setMessages((prev) => [
          ...prev,
          {
            id: makeId('sys'),
            role: 'system',
            text: `요청 실패: ${errorText}`,
          },
        ])
        return
      }

      setLastSql(payload.sql ?? '')
      setLastWarning(payload.warning ?? '')

      const rowCount = Array.isArray(payload.data) ? payload.data.length : 0
      const nextDbCourses =
        !payload.warning && Array.isArray(payload.data)
          ? rowsToCourses(payload.data)
          : null
      setDbCourses(nextDbCourses)
      const summary = payload.warning
        ? 'SQL 생성 완료 (DB 미연결로 결과 조회는 생략되었습니다).'
        : `SQL 실행 완료: ${rowCount}건 결과를 받았습니다.`

      setMessages((prev) => [
        ...prev,
        {
          id: makeId('sys'),
          role: 'system',
          text: summary,
        },
      ])
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        return
      }
      const errorText = '백엔드 연결에 실패했습니다. 서버 실행 상태를 확인해주세요.'
      setLastError(errorText)
      setDbCourses(null)
      setMessages((prev) => [
        ...prev,
        {
          id: makeId('sys'),
          role: 'system',
          text: `요청 실패: ${errorText}`,
        },
      ])
    } finally {
      abortRef.current = null
      setLoading(false)
    }
  }

  if (!mounted) return null

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-end p-4 sm:p-6">
      <button
        type="button"
        aria-label="Close assistant"
        onClick={onClose}
        className={cn(
          'absolute inset-0 bg-slate-900/30 backdrop-blur-[1px] transition-opacity duration-200',
          visible ? 'opacity-100' : 'opacity-0',
        )}
      />

      <div
        className={cn(
          'relative flex max-h-[calc(100vh-2rem)] w-full max-w-md flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl transition duration-200',
          visible ? 'opacity-100 scale-100' : 'opacity-0 scale-95',
        )}
        role="dialog"
        aria-modal="true"
        aria-label="AI Course Assistant"
      >
        <div className="shrink-0 flex items-center justify-between border-b border-slate-200 bg-white px-4 py-3">
          <div>
            <div className="text-sm font-semibold text-slate-900">
              AI Course Assistant
            </div>
            <div className="text-xs text-slate-500">
              Course search · timetable preview
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="inline-flex h-9 w-9 items-center justify-center rounded-lg text-slate-500 transition hover:bg-slate-100 hover:text-slate-700"
            aria-label="Close"
          >
            <span className="text-lg leading-none">×</span>
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto">
          <div className="px-4 pt-4">
            <div
              ref={scrollRef}
              className="h-52 overflow-y-auto rounded-xl border border-slate-200 bg-slate-50 p-3"
            >
              <div className="space-y-2">
                {messages.map((m) => (
                  <div
                    key={m.id}
                    className={cn(
                      'flex',
                      m.role === 'user' ? 'justify-end' : 'justify-start',
                    )}
                  >
                    <div
                      className={cn(
                        'max-w-[85%] rounded-2xl px-3 py-2 text-sm shadow-sm',
                        m.role === 'user'
                          ? 'bg-blue-600 text-white'
                          : 'bg-white text-slate-800 ring-1 ring-slate-200',
                      )}
                    >
                      {m.text}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="px-4 py-3">
            <div className="flex items-center gap-2">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') send()
                }}
                placeholder="e.g. 데이터베이스, 네트워크, ML..."
                disabled={loading}
                className="h-10 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm text-slate-900 shadow-sm outline-none transition focus:border-blue-400 focus:ring-4 focus:ring-blue-100"
              />
              <button
                type="button"
                onClick={send}
                className={cn(
                  'h-10 shrink-0 rounded-xl px-4 text-sm font-semibold text-white shadow-sm transition',
                  loading
                    ? 'bg-rose-600 hover:bg-rose-700 active:bg-rose-700'
                    : 'bg-blue-600 hover:bg-blue-700 active:bg-blue-700',
                )}
              >
                {loading ? 'Cancel' : 'Send'}
              </button>
            </div>
            {lastWarning && (
              <div className="mt-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
                {lastWarning}
              </div>
            )}
            {lastError && (
              <div className="mt-2 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-700">
                {lastError}
              </div>
            )}
            {lastSql && (
              <div className="mt-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                <div className="mb-1 text-[11px] font-semibold text-slate-600">
                  Generated SQL
                </div>
                <pre className="overflow-x-auto whitespace-pre-wrap break-all text-[11px] text-slate-700">
                  {lastSql}
                </pre>
              </div>
            )}
          </div>

          {dbCourses !== null && (
            <div className="border-t border-slate-200 px-4 py-3">
              <div className="flex items-center justify-between">
                <div className="text-xs font-semibold text-slate-700">
                  Course results
                </div>
                <div className="text-xs text-slate-500">
                  {results.length} items
                </div>
              </div>
              <div className="mt-2 max-h-72 min-h-36 space-y-2 overflow-y-auto pr-1">
                {results.map((c) => {
                  const already = selectedIds.has(c.id)
                  const hasConflict = getConflictingCourseIds(c, selectedCourses).size > 0
                  const closed = c.status === 'Closed'
                  return (
                    <div
                      key={c.id}
                      onMouseEnter={() => setHoveredCourse(c)}
                      onMouseLeave={() => setHoveredCourse(null)}
                      className="flex items-center justify-between gap-3 rounded-xl border border-slate-200 bg-white px-3 py-2"
                      title={detailTitle(c)}
                    >
                      <div className="min-w-0">
                        {c.departmentName ? (
                          <div
                            className={cn(
                              'mb-1 truncate text-[11px] font-semibold',
                              departmentColor({
                                selected: already,
                                hasConflict,
                                closed,
                              }),
                            )}
                          >
                            {c.departmentName}
                          </div>
                        ) : null}
                        <div className="truncate text-sm font-semibold text-slate-900">
                          {c.name} - {c.id}
                        </div>
                        <div className="truncate text-xs text-slate-500">
                          {c.professor} · {c.timeText}
                          {c.locationText ? ` · Location: ${c.locationText}` : ''}
                        </div>
                      </div>
                      <button
                        type="button"
                        onMouseEnter={() => setHoveredCourse(c)}
                        onFocus={() => setHoveredCourse(c)}
                        onBlur={() => setHoveredCourse(null)}
                        onClick={() => {
                          if (already) {
                            onRemoveCourse(c.id)
                          } else if (hasConflict) {
                            onReplaceCourse(c)
                          } else {
                            onAddCourse(c)
                          }
                        }}
                        className={cn(
                          'h-9 shrink-0 rounded-lg px-3 text-xs font-semibold transition',
                          already
                            ? 'bg-rose-50 text-rose-700 hover:bg-rose-100 hover:text-rose-800'
                            : hasConflict
                              ? 'bg-amber-100 text-amber-800 hover:bg-amber-200'
                              : 'bg-blue-50 text-blue-700 hover:bg-blue-100',
                        )}
                      >
                        {already ? 'Remove' : hasConflict ? 'Replace' : 'Add'}
                      </button>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          <div className="border-t border-slate-200 bg-white px-4 py-4">
            <div className="mb-2 flex items-center justify-between">
              <div className="text-xs font-semibold text-slate-700">
                Mini timetable
              </div>
              <div className="text-xs text-slate-500">
                {selectedCourses.length} selected
              </div>
            </div>
            <Timetable
              courses={selectedCourses}
              variant="mini"
              overlayCourse={hoveredCourse}
              conflictIds={hoveredConflictIds}
              onClick={onOpenExpandedTimetable}
            />
          </div>
        </div>
      </div>
    </div>
  )
}

