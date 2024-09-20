from flask import (
    Flask, send_from_directory, request, render_template, redirect, url_for, 
    flash
)
import os
from RenderAnimation.render_app import annotations_to_animation
from CompatibilityChecker.model_app import models

app = Flask(__name__)
app.secret_key = '_5#y2L"F4'


@app.route('/', methods=['GET', 'POST'])
def home() -> str:
    """
    Renders the home page for file upload. Handles POST request to upload
    and process an image file.
    """
    message = ''
    if request.method == 'POST':
        file = request.files['file']
        if file:
            os.makedirs('characterfiles/image/', exist_ok=True)
            img_path = 'characterfiles/image/' + file.filename
            file.save(img_path)
            flag = models(img_path)

            if flag == 0:
                message = (
                    "No humanoid figure detected. "
                    "Please upload a different image."
                )
                flash(message)
            else:
                return redirect(url_for('index'))
    return render_template('imageinput.html', message=message)


@app.route('/options', methods=['GET', 'POST'])
def index() -> str:
    """
    Renders the options page where users select animation configurations.
    """
    return render_template('options.html')


@app.route('/submit', methods=['POST'])
def submit() -> str:
    """
    Handles form submission for animation options and starts the 
    animation process.
    """
    selected_option = request.form.get('option')
    annotations_to_animation(selected_option)
    return redirect(url_for('output'))


@app.route('/output')
def output() -> str:
    """
    Serves the generated animation (GIF) to the user after processing.
    """
    return send_from_directory('characterfiles/', 'video.gif')


if __name__ == '__main__':
    app.run(debug=True)
