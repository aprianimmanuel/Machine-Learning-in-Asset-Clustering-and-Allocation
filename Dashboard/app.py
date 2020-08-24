#dashboard for Portfolio Selection and Allocation

from flask import Flask, render_template, request

app= Flask(__name__)

#halaman home
@app.route('/')

def home():
    return render_template('home.html')

#halaman dataset
@app.route('/database',methods=['POST','GET'])
def dataset():
    return render_template('dataset.html')

#halaman visualisasi
@app.route('/visualize',methods=['POST','GET'])
def visual():
    return render_template('plot.html')

if __name__=='__main__':
    app.run(debug=True)