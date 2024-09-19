from flask import Flask, send_from_directory, request, render_template, redirect, url_for, flash
import os
from celery import Celery
from RenderAnimation.render_app import annotations_to_animation
from CompatibilityChecker.model_app import models

app = Flask(__name__)

# Configure Celery to use Redis
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

@app.route('/', methods=['GET', 'POST'])
def home():
    message = ''
    if request.method == 'POST':
        file = request.files['file']
        if file:
            os.makedirs('characterfiles/image/', exist_ok=True)
            img_path = 'characterfiles/image/' + file.filename
            file.save(img_path)
            flag = models(img_path)

            if flag == 0:
                message = "No humanoid figure detected. Please upload a different image."
                flash(message)
            else:
                return redirect(url_for('index'))
    return render_template('imageinput.html', message=message)

@app.route('/options', methods=['GET', 'POST'])
def index():
    return render_template('options.html')

# Celery task
@celery.task
def animation_creation(user_choice):
    print("task started")
    # Perform the annotation to animation function
    annotations_to_animation(user_choice)
    # Return a value if needed (this could be anything relevant to the task)
    return "Animation process completed"

@app.route('/submit', methods=['POST'])
def submit():
    selected_option = request.form.get('option')
    # Start the background task using Celery
    task = animation_creation.apply_async(args=[selected_option])
    
    # Redirect to a loading page or show progress while the task runs
    return redirect(url_for('loading', task_id=task.id))

@app.route('/loading/<task_id>')
def loading(task_id):
    # Get the task result from Celery
    task = animation_creation.AsyncResult(task_id)
    
    if task.state == 'SUCCESS':
        # Redirect to the output page
        return redirect(url_for('output'))
    elif task.state == 'FAILURE':
        return f"Task failed: {task.info}", 500
    else:
        # Task is still processing, render the page with meta refresh
        return render_template('loading.html', task_id=task_id)
    
@app.route('/output')
def output():
    # Serve the GIF file (assuming video.gif is the output)
    return send_from_directory('characterfiles/', 'video.gif')

if __name__ == '__main__':
    app.run(debug=True)
