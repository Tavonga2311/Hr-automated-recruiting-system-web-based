import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SubmitField, MultipleFileField
from wtforms.validators import DataRequired
import pdfminer.high_level
from docx import Document
import magic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import sqlite3
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx'}
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cv_analysis.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database Models
class JobAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    job_title = db.Column(db.String(200), nullable=False)
    job_description = db.Column(db.Text, nullable=False)
    required_skills = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Candidate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    job_analysis_id = db.Column(db.Integer, db.ForeignKey('job_analysis.id'), nullable=False)
    filename = db.Column(db.String(200), nullable=False)
    cv_text = db.Column(db.Text, nullable=False)
    total_score = db.Column(db.Float, nullable=False)
    jd_match = db.Column(db.Float, nullable=False)
    skills_match = db.Column(db.Float, nullable=False)
    found_skills = db.Column(db.Text)
    missing_skills = db.Column(db.Text)
    analyzed_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    job_analysis = db.relationship('JobAnalysis', backref=db.backref('candidates', lazy=True))

class RecruitmentForm(FlaskForm):
    job_title = StringField('Job Position', validators=[DataRequired()])
    job_description = TextAreaField('Job Description', validators=[DataRequired()])
    required_skills = TextAreaField('Required Skills (comma separated)', validators=[DataRequired()])
    cv_files = MultipleFileField('Upload CVs (Multiple)', validators=[DataRequired()])
    submit = SubmitField('Analyze CVs')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(pdf_path):
    return pdfminer.high_level.extract_text(pdf_path)

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_file(file_path):
    mime = magic.Magic(mime=True)
    file_type = mime.from_file(file_path)
    
    if file_type == 'application/pdf':
        return extract_text_from_pdf(file_path)
    elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file type")

def normalize_skill(skill):
    skill = re.sub(r'[^\w\s-]', '', skill.lower())
    skill = re.sub(r'\s+', ' ', skill).strip()
    return skill

def calculate_match(cv_text, job_description, required_skills):
    cv_clean = re.sub(r'[^\w\s]', '', cv_text.lower())
    jd_clean = re.sub(r'[^\w\s]', '', job_description.lower())
    
    skills_list = [normalize_skill(skill) for skill in required_skills.split(',') if skill.strip()]
    
    # Calculate JD match (30% weight)
    jd_match = 0
    if jd_clean:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([jd_clean, cv_clean])
        jd_match = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    # Calculate skills match (70% weight)
    found_skills = []
    missing_skills = []
    
    for skill in skills_list:
        pattern = r'(^|\s)' + re.escape(skill) + r'(\s|$)'
        if re.search(pattern, cv_clean):
            found_skills.append(skill)
        else:
            missing_skills.append(skill)
    
    skills_match = len(found_skills) / len(skills_list) if skills_list else 0
    
    # Combined score with 70% skills and 30% JD
    total_score = (skills_match * 0.7) + (jd_match * 0.3)
    
    return {
        'total_score': round(total_score * 100, 2),
        'jd_match': round(jd_match * 100, 2),
        'skills_match': round(skills_match * 100, 2),
        'missing_skills': ', '.join(missing_skills),
        'found_skills': ', '.join(found_skills),
        'cv_text': cv_text
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    form = RecruitmentForm()
    previous_analyses = JobAnalysis.query.order_by(JobAnalysis.created_at.desc()).all()
    
    if form.validate_on_submit():
        if 'cv_files' not in request.files:
            flash('No files selected')
            return redirect(request.url)
        
        files = request.files.getlist('cv_files')
        if not files or all(file.filename == '' for file in files):
            flash('No files selected')
            return redirect(request.url)
        
        # Create new job analysis record
        job_analysis = JobAnalysis(
            job_title=form.job_title.data,
            job_description=form.job_description.data,
            required_skills=form.required_skills.data
        )
        db.session.add(job_analysis)
        db.session.commit()
        
        candidates = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                try:
                    cv_text = extract_text_from_file(filepath)
                    analysis = calculate_match(
                        cv_text,
                        form.job_description.data,
                        form.required_skills.data
                    )
                    
                    # Store candidate in database
                    candidate = Candidate(
                        job_analysis_id=job_analysis.id,
                        filename=filename,
                        cv_text=cv_text,
                        total_score=analysis['total_score'],
                        jd_match=analysis['jd_match'],
                        skills_match=analysis['skills_match'],
                        found_skills=analysis['found_skills'],
                        missing_skills=analysis['missing_skills']
                    )
                    db.session.add(candidate)
                    
                    candidates.append({
                        'filename': filename,
                        'analysis': analysis,
                        'cv_preview': cv_text[:300] + "..." if len(cv_text) > 300 else cv_text
                    })
                except Exception as e:
                    flash(f"Error processing {filename}: {str(e)}")
                finally:
                    if os.path.exists(filepath):
                        os.remove(filepath)
            else:
                flash(f"Invalid file type: {file.filename}")
        
        db.session.commit()
        
        if not candidates:
            flash("No valid CVs processed")
            return redirect(request.url)
        
        # Sort candidates by total score (descending)
        candidates.sort(key=lambda x: x['analysis']['total_score'], reverse=True)
        
        return render_template('results.html', 
                            job_title=form.job_title.data,
                            candidates=candidates,
                            best_candidate=candidates[0] if candidates else None)
    
    return render_template('index.html', form=form, previous_analyses=previous_analyses)

@app.route('/analysis/<int:analysis_id>')
def view_analysis(analysis_id):
    job_analysis = JobAnalysis.query.get_or_404(analysis_id)
    candidates = Candidate.query.filter_by(job_analysis_id=analysis_id)\
                              .order_by(Candidate.total_score.desc())\
                              .all()
    
    # Convert to the same format as new analyses for the template
    candidate_results = []
    for candidate in candidates:
        candidate_results.append({
            'filename': candidate.filename,
            'analysis': {
                'total_score': candidate.total_score,
                'jd_match': candidate.jd_match,
                'skills_match': candidate.skills_match,
                'missing_skills': candidate.missing_skills.split(', ') if candidate.missing_skills else [],
                'found_skills': candidate.found_skills.split(', ') if candidate.found_skills else []
            },
            'cv_preview': candidate.cv_text[:300] + "..." if len(candidate.cv_text) > 300 else candidate.cv_text
        })
    
    return render_template('results.html',
                         job_title=job_analysis.job_title,
                         candidates=candidate_results,
                         best_candidate=candidate_results[0] if candidate_results else None,
                         is_historical=True)

@app.cli.command('initdb')
def init_db():
    """Initialize the database."""
    db.create_all()
    print('Initialized the database.')

if __name__ == '__main__':
    app.run(debug=True)