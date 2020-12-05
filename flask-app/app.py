from flask import Flask, render_template, request, send_file
import sys
sys.path.insert(0, './model') 
from model import *

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def home():
  translation = ""
  attention = None
  if request.method == "POST":
    translation =  translation_model.translate(request.form['sentence']);
    attention = './attention.png'
  
  return render_template('home.html', data=translation)

@app.route('/attention', methods = ['GET'])
def attention():
  return send_file('attention.png', mimetype='image/png')

@app.after_request
def add_header(response):
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

if __name__ == '__main__':
  app.debug = True
  port = int(os.environ.get("PORT", 5000))
  app.run(host='0.0.0.0', port=port)