export const courseColors = [
  'bg-blue-600/90 ring-blue-200',
  'bg-emerald-600/90 ring-emerald-200',
  'bg-violet-600/90 ring-violet-200',
  'bg-rose-600/90 ring-rose-200',
  'bg-cyan-600/90 ring-cyan-200',
  'bg-orange-500/90 ring-orange-200',
  'bg-fuchsia-600/90 ring-fuchsia-200',
  'bg-teal-600/90 ring-teal-200',
  'bg-lime-600/90 ring-lime-200',
  'bg-indigo-600/90 ring-indigo-200',
]

export function courseColor(index: number) {
  return courseColors[index % courseColors.length]
}
