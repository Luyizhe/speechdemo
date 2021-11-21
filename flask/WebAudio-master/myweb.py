from flask import Flask, render_template, request
import speech_recognition  as sr
from werkzeug.utils import secure_filename
UPLOAD_FOLDER = 'E:\\NLP\SpeechEmotion\demo\flask\WebAudio-master\tmp'
import io, os

app = Flask(__name__,  )
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/')
def index():
	return render_template('index.html')

@app.route("/js_call", methods=['GET','POST'])
def transcriptionAudioToText():
    speech = ''
    if request.method =='POST':
        fname = request.files['audio_data'].filename
        blob = request.files['audio_data']
        name = fname + '.wav'
        filename = secure_filename(name)
        blob.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))


        r = sr.Recognizer()
        audio_file = './upload_files/' + filename
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source)

        try:
            speech = r.recognize_google(audio, language = 'es-ES')
            print(speech)
        except sr.RequestError as e:
            print('No se han podido recuperar los datos del servicio, {0}'.format(e))
        except:
            print('No se ha podido reconocer el audio.')
        os.remove(audio_file)
    return speech

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=7777, debug=True)
