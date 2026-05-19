export type Weekday = 'Mon' | 'Tue' | 'Wed' | 'Thu' | 'Fri' | 'Sat'

export type CourseTimeSlot = {
  day: Weekday
  startHour: number
  endHour: number
}

export type CourseDetail = {
  label: string
  value: string
}

export type Course = {
  id: string
  name: string
  departmentName?: string
  professor: string
  credits: number
  status: 'Open' | 'Closed' | 'Waitlist'
  capacity?: number
  enrolled?: number
  timeText: string
  locationText?: string
  slots: CourseTimeSlot[]
  details?: CourseDetail[]
}

