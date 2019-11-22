from flask import Flask, render_template, request
from inference import get_prediction
app = Flask(__name__)

import os
import time
CONTENT_FOLDER = os.path.join('static', 'target')
if not os.path.exists(CONTENT_FOLDER):
    os.mkdir(CONTENT_FOLDER)
app.config['UPLOAD_FOLDER'] = CONTENT_FOLDER

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # if not request.files.get('content', None) or not request.files.get('style', None):
        #     return redirect(request.url)

        content = request.files.get('content')
        style = request.files.get('style')

        target = get_prediction(content, style)
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], "target.png")
        target.save(full_filename)
        while not os.path.isfile(full_filename):
            print("in")
            time.sleep(1)
        return render_template("result.html", image=full_filename)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
