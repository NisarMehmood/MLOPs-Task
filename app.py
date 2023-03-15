from flask import Flask, render_template, request
import pickle

app = Flask(__name__, template_folder='template')

@app.route('/')
def home():
    return render_template('task.html')


@app.route('/result', methods = ['POST'])
def result():
    output = request.form.to_dict()
    years = output['years']

    model = pickle.load(open('model.pkl', 'rb'))
    result = model.predict([[float(years)]])
    result = result[0]
    return render_template('task.html', result = result)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')