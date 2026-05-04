# PhysioTrack - Patient Exercise Schedule Manager

## Concept & Vision

A streamlined web application for physiotherapists to manage patient exercise schedules with printable, patient-friendly handouts. Eliminates the 100-page Excel chaos with clean, focused outputs—one patient, one document, perfect for printing. Clinical but warm, professional but approachable.

## Design Language

- **Aesthetic**: Clean clinical UI with soft, calming touches. Think modern healthcare app—not sterile, but trustworthy.
- **Colors**:
  - Primary: `#2563eb` (confident blue)
  - Secondary: `#10b981` (healing green)
  - Accent: `#f59e0b` (warm amber)
  - Background: `#f8fafc`
  - Card: `#ffffff`
  - Text: `#1e293b`
  - Muted: `#64748b`
- **Typography**: Inter (clean, modern sans-serif)
- **Spatial system**: Generous whitespace, card-based layouts, 8px grid
- **Motion**: Subtle, functional—200ms ease transitions on interactions
- **Icons**: Lucide icons (consistent stroke weight)

## Layout & Structure

### Main Sections
1. **Dashboard**: Patient list with quick stats (active patients, recent additions)
2. **Patient View**: Individual patient with all their exercise programs
3. **Exercise Library**: Reusable exercises that can be assigned to patients
4. **Print View**: Optimized single-page-per-patient format for printing

### Responsive Strategy
- Desktop-first (physiotherapist workflow)
- Print stylesheet for clean A4/Letter output

## Features & Interactions

### Patient Management
- Add/edit/archive patients
- Patient details: name, contact, notes, start date
- Search and filter patients

### Exercise Program Builder
- Create named programs (e.g., "Post-Surgery Week 1-2")
- Drag-and-drop exercise ordering
- Set reps, sets, hold time, notes per assignment
- Assign programs to patients with date ranges

### Exercise Library
- Pre-built exercise catalog with categories (mobility, strength, balance, etc.)
- Custom exercise creation
- Each exercise has: name, description, difficulty, equipment needed

### Print Mode
- One-click print for individual patient programs
- Clean, high-contrast layout optimized for paper
- Large text, numbered steps, minimal graphics
- No navigation—just the exercises

## Component Inventory

### PatientCard
- Avatar with initials, name, last active date
- Program count badge
- Quick actions (view, print, edit)
- States: default, hover (lift shadow), selected

### ExerciseItem
- Name, category tag, difficulty indicator
- Assigned reps/sets in context
- Drag handle for reordering
- States: default, dragging (elevated shadow), hover

### ProgramCard
- Program name, date range, exercise count
- Expand/collapse to show exercises
- Edit/delete actions
- States: collapsed, expanded, hover

### PrintDocument
- Patient header with contact info
- Large exercise name
- Clear instructions (reps × sets, hold time)
- Equipment icons
- Clean numbered list format

## Technical Approach

- **Framework**: Flask with SQLite (simple, no setup required)
- **Frontend**: Vanilla HTML/CSS/JS with Tailwind CDN
- **Database**: SQLite via SQLAlchemy
- **Print**: CSS @media print stylesheet
- **No build step**: Run with `python app.py`

## Data Model

### Patient
- id, name, email, phone, notes, created_at, is_active

### Exercise
- id, name, description, category, difficulty, equipment, instructions

### Program
- id, patient_id, name, start_date, end_date, notes

### ProgramExercise (join table)
- id, program_id, exercise_id, order, reps, sets, hold_seconds, notes