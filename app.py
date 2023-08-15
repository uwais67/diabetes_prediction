from flask import Flask , render_template, request
import pickle
app=Flask(__name__)
clf=pickle.load(open("model/classifier.pkl",'rb'))

@app.route('/', methods=["GET","POST"])
def home():
      pred=None
      if request.method == 'POST':
        a1 = request.form.get('entry1')
        a2 = request.form.get('entry2')
        a3 = request.form.get('entry3')
        a4 = request.form.get('entry4')
        a5 = request.form.get('entry5')
        a6 = request.form.get('entry6')
        a7 = request.form.get('entry7')
        a8 = request.form.get('entry8')
        pred=clf.predict([[a1,a2,a3,a4,a5,a6,a7,a8]])
        pred=pred[0]
      return render_template('index.html', pred=pred)

if __name__=="__main__":
    app.run(debug=True)