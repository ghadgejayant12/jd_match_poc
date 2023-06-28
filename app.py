from flask import Flask, session, render_template, request, send_file
#from inference import MakeInference
from inference2 import MakeInference
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        path = request.form.get('resume_path')
        jd = request.form.get('job_description')
        inf = MakeInference(path)
        print('Using :', jd)
        inf.load_docs()
        print('here')
        result = inf.rank_jd(jd_text=jd)
        result.sort(reverse=True)
        for i in range(len(result)):
            print(result[i])
            result[i].append(i)
        return render_template('home.html', data=True, result=result)
    else:
        return render_template('home.html')


@app.route('/download', methods=['GET', 'POST'])
def download_view():
    filepath = request.args.get('filepath')
    print('This is the path :', filepath)
    return send_file(filepath, as_attachment=False)


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
