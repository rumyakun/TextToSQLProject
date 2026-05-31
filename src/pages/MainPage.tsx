import { useEffect, useMemo, useState } from 'react'
import ChatPopup from '../components/ChatPopup'
import CourseTable from '../components/CourseTable'
import ExpandedTimetableModal from '../components/ExpandedTimetableModal'
import Timetable from '../components/Timetable'
import { coursesApi } from '../services/api'
import type { Course } from '../types/course'
import { cn } from '../utils/cn'
import { courseColor } from '../utils/courseColors'
import { apiCourseToCourse } from '../utils/courseMapper'
import {
  extractCompletedCourseCodes,
  formatPrerequisites,
  getUnmetPrerequisiteCodes,
  getUnmetPrerequisiteNames,
} from '../utils/prerequisites'
import { getConflictingCourseIds } from '../utils/schedule'

function ChatIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <path
        d="M7.5 20.5c1.6.5 3.2.8 4.5.8 5.2 0 9-3.6 9-8.4S17.2 4.5 12 4.5 3 8.1 3 12.9c0 1.9.6 3.6 1.8 5.1l-.6 3.2 3.1-.9c.4.2.8.4 1.2.6Z"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinejoin="round"
      />
      <path
        d="M7.7 12.7h.8m3.1 0h.8m3.1 0h.8"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
    </svg>
  )
}

type MainPageProps = {
  userStudentNo?: string
  userName?: string
  completedCourses?: unknown[]
  onLoginClick: () => void
  onLogout: () => void
}

function buildCoursePrompt(
  course: Course,
  action: 'add' | 'replace',
  completedCourseCodes: Set<string> | null,
) {
  const closed =
    typeof course.capacity === 'number' &&
    typeof course.enrolled === 'number' &&
    course.capacity > 0 &&
    course.enrolled >= course.capacity
  const unmetPrerequisiteCodes = getUnmetPrerequisiteCodes(course, completedCourseCodes)

  if (!closed && unmetPrerequisiteCodes.length === 0) return null

  return {
    course,
    action,
    closed,
    unmetPrerequisiteCodes,
    unmetPrerequisiteNames: getUnmetPrerequisiteNames(course, unmetPrerequisiteCodes),
  }
}

export default function MainPage({
  userStudentNo,
  userName,
  completedCourses,
  onLoginClick,
  onLogout,
}: MainPageProps) {
  const [allCourses, setAllCourses] = useState<Course[]>([])
  const [popupOpen, setPopupOpen] = useState(false)
  const [expandedOpen, setExpandedOpen] = useState(false)
  const [selectedCourses, setSelectedCourses] = useState<Course[]>([])
  const [hoveredCourse, setHoveredCourse] = useState<Course | null>(null)
  const [courseSearch, setCourseSearch] = useState('')
  const [coursePrompt, setCoursePrompt] = useState<{
    course: Course
    action: 'add' | 'replace'
    closed: boolean
    unmetPrerequisiteCodes: string[]
    unmetPrerequisiteNames: Array<string | undefined>
  } | null>(null)

  useEffect(() => {
    let disposed = false
    const timeoutId = window.setTimeout(() => {
      void loadCourses()
    }, 250)

    async function loadCourses() {
      try {
        const keyword = courseSearch.trim()
        const result = await coursesApi.getCourses({
          year: new Date().getFullYear(),
          semester: '1',
          keyword: keyword || undefined,
          page: 1,
          pageSize: 200,
        })
        const courses = result.items.map(apiCourseToCourse)
        if (!disposed && courses.length > 0) {
          setAllCourses(courses)
        }
      } catch {
        if (!disposed) {
          setAllCourses([])
        }
      }
    }

    return () => {
      disposed = true
      window.clearTimeout(timeoutId)
    }
  }, [courseSearch])

  const selectedIds = useMemo(
    () => new Set(selectedCourses.map((c) => c.id)),
    [selectedCourses],
  )
  const hoveredConflictIds = useMemo(
    () => getConflictingCourseIds(hoveredCourse, selectedCourses),
    [hoveredCourse, selectedCourses],
  )
  const completedCourseCodes = useMemo(
    () => (userStudentNo ? extractCompletedCourseCodes(completedCourses) : null),
    [completedCourses, userStudentNo],
  )

  function addCourseDirect(course: Course) {
    setSelectedCourses((prev) => {
      if (prev.some((c) => c.id === course.id)) return prev
      return [...prev, course]
    })
  }

  function replaceConflictingCoursesDirect(course: Course) {
    setSelectedCourses((prev) => {
      const conflicts = getConflictingCourseIds(course, prev)
      const next = prev.filter((c) => c.id !== course.id && !conflicts.has(c.id))
      return [...next, course]
    })
  }

  function addCourse(course: Course) {
    const warning = buildCoursePrompt(course, 'add', completedCourseCodes)
    if (warning) {
      setCoursePrompt(warning)
      return
    }
    addCourseDirect(course)
  }

  function removeCourse(courseId: string) {
    setSelectedCourses((prev) => prev.filter((c) => c.id !== courseId))
  }

  function replaceConflictingCourses(course: Course) {
    const warning = buildCoursePrompt(course, 'replace', completedCourseCodes)
    if (warning) {
      setCoursePrompt(warning)
      return
    }
    replaceConflictingCoursesDirect(course)
  }

  function confirmCoursePrompt() {
    if (!coursePrompt) return
    if (coursePrompt.action === 'replace') {
      replaceConflictingCoursesDirect(coursePrompt.course)
    } else {
      addCourseDirect(coursePrompt.course)
    }
    setCoursePrompt(null)
  }

  return (
    <div className="min-h-screen bg-slate-50">
      {coursePrompt ? (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-slate-900/35 px-4">
          <div
            className="w-full max-w-sm rounded-xl border border-slate-200 bg-white p-5 shadow-2xl"
            role="dialog"
            aria-modal="true"
            aria-labelledby="course-warning-title"
          >
            <div id="course-warning-title" className="text-sm font-semibold text-slate-900">
              확인이 필요한 강의입니다
            </div>
            <div className="mt-2 text-sm text-slate-600">
              {coursePrompt.course.name}을(를) 시간표에 담기 전에 확인해 주세요.
            </div>
            <div className="mt-3 space-y-2">
              {coursePrompt.closed ? (
                <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs font-medium text-amber-900">
                  정원이 마감되었거나 초과된 강의입니다.
                </div>
              ) : null}
              {coursePrompt.unmetPrerequisiteCodes.length > 0 ? (
                <div className="rounded-lg border border-violet-200 bg-violet-50 px-3 py-2 text-xs font-medium text-violet-900">
                  선수과목을 아직 이수하지 않았습니다:
                  <span className="ml-1">
                    {formatPrerequisites(
                      coursePrompt.unmetPrerequisiteCodes,
                      coursePrompt.unmetPrerequisiteNames,
                    )}
                  </span>
                </div>
              ) : null}
            </div>
            <div className="mt-4 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setCoursePrompt(null)}
                className="h-9 rounded-lg border border-slate-200 bg-white px-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
              >
                취소
              </button>
              <button
                type="button"
                onClick={confirmCoursePrompt}
                className="h-9 rounded-lg bg-violet-700 px-3 text-sm font-semibold text-white transition hover:bg-violet-800"
              >
                그래도 담기
              </button>
            </div>
          </div>
        </div>
      ) : null}

      <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/80 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center gap-4 px-4 py-4 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-600 text-white shadow-sm">
              <span className="text-sm font-black">UNI</span>
            </div>
            <div>
              <div className="text-sm font-semibold text-slate-900">
                Course Registration
              </div>
              <div className="text-xs text-slate-500">
                Main course browser
              </div>
            </div>
          </div>

          <div className="ml-auto w-full max-w-xl">
            <div className="relative">
              <input
                value={courseSearch}
                onChange={(event) => setCourseSearch(event.target.value)}
                placeholder="Search by course name"
                className="h-11 w-full rounded-xl border border-slate-200 bg-white px-4 pr-12 text-sm text-slate-900 shadow-sm outline-none transition focus:border-blue-400 focus:ring-4 focus:ring-blue-100"
              />
              <div className="pointer-events-none absolute inset-y-0 right-3 flex items-center text-slate-400">
                <svg
                  width="18"
                  height="18"
                  viewBox="0 0 24 24"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                  aria-hidden="true"
                >
                  <path
                    d="M10.5 18a7.5 7.5 0 1 1 0-15 7.5 7.5 0 0 1 0 15Z"
                    stroke="currentColor"
                    strokeWidth="1.8"
                  />
                  <path
                    d="M16.2 16.2 21 21"
                    stroke="currentColor"
                    strokeWidth="1.8"
                    strokeLinecap="round"
                  />
                </svg>
              </div>
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            {userName ? (
              <>
                <div className="hidden rounded-lg bg-slate-100 px-3 py-2 text-xs font-medium text-slate-700 sm:block">
                  {userName}
                </div>
                <button
                  type="button"
                  onClick={onLogout}
                  className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 transition hover:bg-slate-50"
                >
                  Logout
                </button>
              </>
            ) : (
              <button
                type="button"
                onClick={onLoginClick}
                className="rounded-lg bg-blue-600 px-3 py-2 text-xs font-semibold text-white transition hover:bg-blue-700"
              >
                Login
              </button>
            )}
          </div>
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl grid-cols-1 gap-5 px-4 py-6 sm:px-6 lg:grid-cols-[1.45fr_0.95fr]">
        <section className="min-w-0">
          <CourseTable
            courses={allCourses}
            selectedIds={selectedIds}
            selectedCourses={selectedCourses}
            completedCourseCodes={completedCourseCodes}
            onAddCourse={addCourse}
            onRemoveCourse={removeCourse}
            onHoverCourse={setHoveredCourse}
            onReplaceCourse={replaceConflictingCourses}
          />
        </section>

        <aside className="space-y-5">
          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm font-semibold text-slate-900">
                  Selected Courses
                </div>
                <div className="text-xs text-slate-500">
                  Added via the assistant popup
                </div>
              </div>
              <div className="rounded-full bg-blue-50 px-3 py-1 text-xs font-semibold text-blue-700">
                {selectedCourses.length}
              </div>
            </div>

            <div className="mt-3 max-h-64 space-y-2 overflow-y-auto pr-1">
              {selectedCourses.length === 0 ? (
                <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50 px-3 py-3 text-sm text-slate-600">
                  Open the assistant and add a course to build your timetable.
                </div>
              ) : (
                selectedCourses.map((c, index) => (
                  <div
                    key={c.id}
                    className="flex items-center justify-between gap-3 rounded-xl border border-slate-200 bg-white px-3 py-2"
                  >
                    <div className="flex min-w-0 items-start gap-3">
                      <div
                        className={cn(
                          'mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-lg text-xs font-black text-white shadow-sm ring-1',
                          courseColor(index),
                        )}
                      >
                        {index + 1}
                      </div>
                      <div className="min-w-0">
                        <div className="truncate text-sm font-semibold text-slate-900">
                          {c.name}
                        </div>
                        <div className="truncate text-xs text-slate-500">
                          {c.timeText}
                        </div>
                        {c.locationText ? (
                          <div className="truncate text-xs text-slate-500">
                            Location: {c.locationText}
                          </div>
                        ) : null}
                      </div>
                    </div>
                    <div className="flex shrink-0 items-center gap-2">
                      <div className="text-xs font-semibold text-slate-600">
                        {c.credits > 0 ? `${c.credits}cr` : 'Credits TBA'}
                      </div>
                      <button
                        type="button"
                        onClick={() => removeCourse(c.id)}
                        className="inline-flex h-7 w-7 items-center justify-center rounded-lg text-slate-400 transition hover:bg-rose-50 hover:text-rose-700"
                        aria-label={`Remove ${c.name}`}
                      >
                        <span className="text-base leading-none">×</span>
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          <Timetable
            courses={selectedCourses}
            variant="mini"
            overlayCourse={hoveredCourse}
            conflictIds={hoveredConflictIds}
            onClick={() => setExpandedOpen(true)}
          />
        </aside>
      </main>

      <button
        type="button"
        onClick={() => setPopupOpen(true)}
        className={cn(
          'fixed bottom-6 right-6 z-40 inline-flex h-14 w-14 items-center justify-center rounded-full bg-blue-600 text-white shadow-xl',
          'transition hover:bg-blue-700 active:scale-[0.98]',
        )}
        aria-label="Open AI Course Assistant"
      >
        <ChatIcon className="h-6 w-6" />
      </button>

      <ChatPopup
        open={popupOpen}
        onClose={() => setPopupOpen(false)}
        userStudentNo={userStudentNo}
        selectedCourses={selectedCourses}
        selectedIds={selectedIds}
        completedCourseCodes={completedCourseCodes}
        onAddCourse={addCourse}
        onRemoveCourse={removeCourse}
        onReplaceCourse={replaceConflictingCourses}
        onOpenExpandedTimetable={() => setExpandedOpen(true)}
      />

      <ExpandedTimetableModal
        open={expandedOpen}
        courses={selectedCourses}
        onClose={() => setExpandedOpen(false)}
      />
    </div>
  )
}
