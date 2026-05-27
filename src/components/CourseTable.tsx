import type { Course } from '../types/course'
import { cn } from '../utils/cn'
import { getConflictingCourseIds } from '../utils/schedule'

const statusStyles: Record<Course['status'], string> = {
  Open: 'bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200',
  Closed: 'bg-slate-100 text-slate-600 ring-1 ring-slate-200',
  Waitlist: 'bg-amber-50 text-amber-700 ring-1 ring-amber-200',
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

export default function CourseTable({
  courses,
  selectedIds,
  selectedCourses,
  onAddCourse,
  onRemoveCourse,
  onHoverCourse,
  onReplaceCourse,
}: {
  courses: Course[]
  selectedIds: Set<string>
  selectedCourses: Course[]
  onAddCourse: (course: Course) => void
  onRemoveCourse: (courseId: string) => void
  onHoverCourse: (course: Course | null) => void
  onReplaceCourse: (course: Course) => void
}) {
  return (
    <div className="flex max-h-[calc(100vh-8.5rem)] min-h-[28rem] flex-col rounded-2xl border border-slate-200 bg-white shadow-sm">
      <div className="shrink-0 flex items-center justify-between border-b border-slate-200 px-5 py-4">
        <div>
          <div className="text-sm font-semibold text-slate-900">
            Course List
          </div>
          <div className="text-xs text-slate-500">
            {courses.length} courses
          </div>
        </div>
        <div className="text-xs text-slate-500">
          Selected: <span className="font-semibold">{selectedIds.size}</span>
        </div>
      </div>

      <div className="min-h-0 overflow-auto">
        <table className="w-full min-w-[1040px] table-auto">
          <thead className="sticky top-0 z-[1] bg-slate-50 text-left text-xs font-semibold text-slate-600 shadow-[0_1px_0_0_rgba(226,232,240,1)]">
            <tr>
              <th className="px-5 py-3">Action</th>
              <th className="px-5 py-3">Course Name</th>
              <th className="px-5 py-3">Professor</th>
              <th className="px-5 py-3">Time</th>
              <th className="px-5 py-3">Credits</th>
              <th className="px-5 py-3">Enrolled</th>
              <th className="px-5 py-3">Capacity</th>
              <th className="px-5 py-3">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 text-sm">
            {courses.map((c) => {
              const selected = selectedIds.has(c.id)
              const hasConflict = getConflictingCourseIds(c, selectedCourses).size > 0
              const closed = c.status === 'Closed'
              return (
                <tr
                  key={c.id}
                  onMouseEnter={() => onHoverCourse(c)}
                  onMouseLeave={() => onHoverCourse(null)}
                  className={cn(
                    'hover:bg-slate-50/70',
                    selected && 'bg-blue-50/40',
                    hasConflict && !selected && 'bg-amber-50/40',
                  )}
                >
                  <td className="px-5 py-4">
                    <button
                      type="button"
                      onMouseEnter={() => onHoverCourse(c)}
                      onFocus={() => onHoverCourse(c)}
                      onBlur={() => onHoverCourse(null)}
                      onClick={() => {
                        if (selected) {
                          onRemoveCourse(c.id)
                        } else if (hasConflict) {
                          onReplaceCourse(c)
                        } else {
                          onAddCourse(c)
                        }
                      }}
                      className={cn(
                        'h-8 rounded-lg px-3 text-xs font-semibold transition',
                        selected
                          ? 'bg-rose-50 text-rose-700 hover:bg-rose-100 hover:text-rose-800'
                          : hasConflict
                            ? 'bg-amber-100 text-amber-800 hover:bg-amber-200'
                            : 'bg-blue-50 text-blue-700 hover:bg-blue-100',
                      )}
                    >
                      {selected ? 'REMOVE' : hasConflict ? 'REPLACE' : 'ADD'}
                    </button>
                  </td>
                  <td className="px-5 py-4">
                    <div className="flex items-start gap-3">
                      <div className="mt-0.5 h-2.5 w-2.5 rounded-full bg-blue-600/90" />
                      <div className="min-w-0">
                        {c.departmentName ? (
                          <div
                            className={cn(
                              'mb-1 truncate text-[11px] font-semibold',
                              departmentColor({ selected, hasConflict, closed }),
                            )}
                          >
                            {c.departmentName}
                          </div>
                        ) : null}
                        <div className="font-semibold text-slate-900">
                          {c.name}
                        </div>
                        <div className="text-xs text-slate-500">{c.id}</div>
                      </div>
                    </div>
                  </td>
                  <td className="px-5 py-4 text-slate-700">{c.professor}</td>
                  <td className="px-5 py-4 text-slate-700">
                    <div>{c.timeText}</div>
                    {c.locationText ? (
                      <div className="mt-1 text-xs text-slate-500">
                        Location: {c.locationText}
                      </div>
                    ) : null}
                  </td>
                  <td className="px-5 py-4 text-slate-700">
                    {c.credits > 0 ? c.credits : 'TBA'}
                  </td>
                  <td className="px-5 py-4 text-slate-700">
                    {typeof c.enrolled === 'number' ? c.enrolled : 'TBA'}
                  </td>
                  <td className="px-5 py-4 text-slate-700">
                    {typeof c.capacity === 'number' ? c.capacity : 'TBA'}
                  </td>
                  <td className="px-5 py-4">
                    <span
                      className={cn(
                        'inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold',
                        statusStyles[c.status],
                      )}
                    >
                      {c.status}
                    </span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

