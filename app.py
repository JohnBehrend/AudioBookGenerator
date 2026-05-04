from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'physio-track-secret-key-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///physiotrack.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120))
    phone = db.Column(db.String(20))
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    programs = db.relationship('Program', backref='patient', lazy=True)


class Exercise(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    category = db.Column(db.String(50), default='mobility')
    difficulty = db.Column(db.String(20), default='moderate')
    equipment = db.Column(db.String(200))
    instructions = db.Column(db.Text)
    is_active = db.Column(db.Boolean, default=True)


class Program(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    start_date = db.Column(db.Date)
    end_date = db.Column(db.Date)
    notes = db.Column(db.Text)
    program_exercises = db.relationship('ProgramExercise', backref='program', lazy=True, order_by='ProgramExercise.order')


class ProgramExercise(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    program_id = db.Column(db.Integer, db.ForeignKey('program.id'), nullable=False)
    exercise_id = db.Column(db.Integer, db.ForeignKey('exercise.id'), nullable=False)
    exercise = db.relationship('Exercise')
    order = db.Column(db.Integer, default=0)
    reps = db.Column(db.Integer)
    sets = db.Column(db.Integer)
    hold_seconds = db.Column(db.Integer)
    notes = db.Column(db.Text)


def seed_exercises():
    if Exercise.query.count() == 0:
        exercises = [
            Exercise(name="Ankle Pumps", description="Improve circulation and ankle mobility", category="mobility", difficulty="easy", equipment="None", instructions="Lie down or sit with legs extended. Slowly flex and extend your ankle, pointing toes away then toward you. Repeat continuously."),
            Exercise(name="Quad Sets", description="Strengthen quadriceps muscle", category="strength", difficulty="easy", equipment="None", instructions="Sit or lie with leg straight. Tighten the muscle on top of your thigh by pushing the back of your knee down into the floor. Hold and release."),
            Exercise(name="Straight Leg Raise", description="Strengthen hip flexors and quadriceps", category="strength", difficulty="moderate", equipment="None", instructions="Lie on your back with one knee bent. Keep the other leg straight. Lift the straight leg about 12 inches. Hold, then lower slowly."),
            Exercise(name="Heel Slides", description="Improve knee flexion range of motion", category="mobility", difficulty="easy", equipment="None", instructions="Lie on your back. Slowly slide your heel toward your buttock, bending your knee. Slide back to straight position. Repeat."),
            Exercise(name="Hip Abduction", description="Strengthen hip abductors", category="strength", difficulty="moderate", equipment="None", instructions="Lie on your side with legs stacked. Keep top leg straight and lift it toward the ceiling. Hold, then lower slowly."),
            Exercise(name="Hamstring Stretch", description="Stretch the back of the thigh", category="flexibility", difficulty="easy", equipment="None", instructions="Lie on your back. Lift one leg and hold behind the thigh. Gently straighten the knee until you feel a stretch. Hold."),
            Exercise(name="Calf Stretch", description="Stretch calf muscles", category="flexibility", difficulty="easy", equipment="Wall or chair", instructions="Stand facing a wall. Place hands on wall and step one foot back. Keep back heel on floor. Lean forward until you feel a stretch in your calf."),
            Exercise(name="Wall Squats", description="Strengthen quadriceps and improve knee stability", category="strength", difficulty="moderate", equipment="Wall", instructions="Stand with your back against a wall. Slowly slide down until your knees are bent at about 45 degrees. Hold, then slide back up."),
            Exercise(name="Step-Ups", description="Strengthen legs and improve balance", category="strength", difficulty="moderate", equipment="Step or stair", instructions="Stand facing a step. Step up with one foot, then bring the other foot up. Step down in reverse. Repeat, alternating lead leg."),
            Exercise(name="Balance on One Leg", description="Improve static balance", category="balance", difficulty="moderate", equipment="Chair for support", instructions="Stand behind a chair for support. Lift one foot and balance on the other. Progress to放手 support."),
            Exercise(name="Bridging", description="Strengthen glutes and hamstrings", category="strength", difficulty="easy", equipment="None", instructions="Lie on your back with knees bent, feet flat. Lift hips off the floor by squeezing glutes. Hold at the top, then lower slowly."),
            Exercise(name="Terminal Knee Extension", description="Target vastus medialis obliques", category="strength", difficulty="easy", equipment="Resistance band", instructions="Sit with a rolled towel under knee. Press knee down into towel while straightening the leg. Hold, then release."),
            Exercise(name="Clamshells", description="Strengthen hip external rotators", category="strength", difficulty="easy", equipment="None", instructions="Lie on your side with knees bent, feet together. Keep feet together and lift top knee like a clamshell. Hold, then lower."),
            Exercise(name="Plank", description="Core strengthening", category="strength", difficulty="moderate", equipment="Mat", instructions="Lie face down. Prop up on forearms and toes. Keep body in a straight line from head to heels. Hold position."),
            Exercise(name="Seated Knee Extension", description="Strengthen quadriceps", category="strength", difficulty="easy", equipment="Chair", instructions="Sit in a chair with feet flat. Straighten one knee until leg is parallel to floor. Hold, then lower slowly."),
            Exercise(name="Gastrocnemius Stretch", description="Stretch calf muscles", category="flexibility", difficulty="easy", equipment="Wall", instructions="Stand facing wall. Place both hands on wall. Step one foot back, keeping heel pressed down. Lean into wall until stretch is felt."),
            Exercise(name="Piriformis Stretch", description="Stretch deep hip rotators", category="flexibility", difficulty="moderate", equipment="None", instructions="Lie on your back. Cross one ankle over the opposite knee. Pull the bottom leg toward chest until stretch is felt."),
            Exercise(name="Patellar Mobilization", description="Improve kneecap mobility", category="mobility", difficulty="easy", equipment="None", instructions="Sit with leg extended. Gently push kneecap up, then down, then side to side using your fingers. Move slowly and gently."),
            Exercise(name="Bird Dog", description="Core and back strengthening with balance", category="strength", difficulty="moderate", equipment="None", instructions="Start on hands and knees. Extend opposite arm and leg simultaneously. Keep back flat. Hold, then switch sides."),
            Exercise(name="Side-Lying Hip Adduction", description="Strengthen inner thigh", category="strength", difficulty="easy", equipment="None", instructions="Lie on your side with bottom leg bent for support. Keep top leg straight and lift it toward the floor. Hold, then lower."),
        ]
        db.session.add_all(exercises)
        db.session.commit()


@app.route('/')
def index():
    patients = Patient.query.filter_by(is_active=True).order_by(Patient.name).all()
    stats = {
        'total': Patient.query.filter_by(is_active=True).count(),
        'programs': Program.query.join(Patient).filter(Patient.is_active==True).count()
    }
    return render_template('index.html', patients=patients, stats=stats)


@app.route('/patient/new', methods=['POST'])
def new_patient():
    patient = Patient(
        name=request.form['name'],
        email=request.form.get('email'),
        phone=request.form.get('phone'),
        notes=request.form.get('notes')
    )
    db.session.add(patient)
    db.session.commit()
    flash(f'Patient {patient.name} added successfully', 'success')
    return redirect(url_for('index'))


@app.route('/patient/<int:id>')
def patient(id):
    patient = Patient.query.get_or_404(id)
    return render_template('patient.html', patient=patient)


@app.route('/patient/<int:id>/edit', methods=['POST'])
def edit_patient(id):
    patient = Patient.query.get_or_404(id)
    patient.name = request.form['name']
    patient.email = request.form.get('email')
    patient.phone = request.form.get('phone')
    patient.notes = request.form.get('notes')
    db.session.commit()
    flash(f'Patient {patient.name} updated', 'success')
    return redirect(url_for('patient', id=id))


@app.route('/patient/<int:id>/archive', methods=['POST'])
def archive_patient(id):
    patient = Patient.query.get_or_404(id)
    patient.is_active = False
    db.session.commit()
    flash(f'Patient {patient.name} archived', 'info')
    return redirect(url_for('index'))


@app.route('/api/exercises')
def get_exercises():
    exercises = Exercise.query.filter_by(is_active=True).order_by(Exercise.category, Exercise.name).all()
    return jsonify([{
        'id': e.id,
        'name': e.name,
        'description': e.description,
        'category': e.category,
        'difficulty': e.difficulty,
        'equipment': e.equipment,
        'instructions': e.instructions
    } for e in exercises])


@app.route('/exercises')
def exercises():
    exercises = Exercise.query.filter_by(is_active=True).order_by(Exercise.category, Exercise.name).all()
    categories = set(e.category for e in exercises)
    return render_template('exercises.html', exercises=exercises, categories=categories)


@app.route('/exercise/new', methods=['POST'])
def new_exercise():
    exercise = Exercise(
        name=request.form['name'],
        description=request.form.get('description'),
        category=request.form.get('category', 'mobility'),
        difficulty=request.form.get('difficulty', 'moderate'),
        equipment=request.form.get('equipment', ''),
        instructions=request.form.get('instructions', '')
    )
    db.session.add(exercise)
    db.session.commit()
    flash(f'Exercise {exercise.name} added to library', 'success')
    return redirect(url_for('exercises'))


@app.route('/program/new', methods=['POST'])
def new_program():
    program = Program(
        patient_id=request.form['patient_id'],
        name=request.form['name'],
        start_date=datetime.strptime(request.form['start_date'], '%Y-%m-%d').date() if request.form.get('start_date') else None,
        end_date=datetime.strptime(request.form['end_date'], '%Y-%m-%d').date() if request.form.get('end_date') else None,
        notes=request.form.get('notes')
    )
    db.session.add(program)
    db.session.commit()
    flash(f'Program "{program.name}" created', 'success')
    return redirect(url_for('patient', id=program.patient_id))


@app.route('/program/<int:id>/add-exercise', methods=['POST'])
def add_exercise_to_program(id):
    program = Program.query.get_or_404(id)
    max_order = db.session.query(db.func.max(ProgramExercise.order)).filter_by(program_id=id).scalar() or 0
    
    program_exercise = ProgramExercise(
        program_id=id,
        exercise_id=request.form['exercise_id'],
        order=max_order + 1,
        reps=int(request.form.get('reps', 10)) if request.form.get('reps') else None,
        sets=int(request.form.get('sets', 3)) if request.form.get('sets') else None,
        hold_seconds=int(request.form.get('hold_seconds')) if request.form.get('hold_seconds') else None,
        notes=request.form.get('notes')
    )
    db.session.add(program_exercise)
    db.session.commit()
    return redirect(url_for('patient', id=program.patient_id))


@app.route('/program/<int:id>/reorder', methods=['POST'])
def reorder_program(id):
    program = Program.query.get_or_404(id)
    order = request.json.get('order', [])
    for idx, pe_id in enumerate(order):
        pe = ProgramExercise.query.get(pe_id)
        if pe:
            pe.order = idx
    db.session.commit()
    return jsonify({'success': True})


@app.route('/program-exercise/<int:id>/edit', methods=['POST'])
def edit_program_exercise(id):
    pe = ProgramExercise.query.get_or_404(id)
    pe.reps = int(request.form['reps']) if request.form.get('reps') else None
    pe.sets = int(request.form['sets']) if request.form.get('sets') else None
    pe.hold_seconds = int(request.form['hold_seconds']) if request.form.get('hold_seconds') else None
    pe.notes = request.form.get('notes')
    db.session.commit()
    return redirect(url_for('patient', id=pe.program.patient_id))


@app.route('/program-exercise/<int:id>/delete', methods=['POST'])
def delete_program_exercise(id):
    pe = ProgramExercise.query.get_or_404(id)
    patient_id = pe.program.patient_id
    db.session.delete(pe)
    db.session.commit()
    return redirect(url_for('patient', id=patient_id))


@app.route('/program/<int:id>/delete', methods=['POST'])
def delete_program(id):
    program = Program.query.get_or_404(id)
    patient_id = program.patient_id
    ProgramExercise.query.filter_by(program_id=id).delete()
    db.session.delete(program)
    db.session.commit()
    return redirect(url_for('patient', id=patient_id))


@app.route('/print/patient/<int:id>')
def print_patient(id):
    patient = Patient.query.get_or_404(id)
    return render_template('print.html', patient=patient)


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        seed_exercises()
    print("\n" + "="*50)
    print("  PhysioTrack is running!")
    print("  Open: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)