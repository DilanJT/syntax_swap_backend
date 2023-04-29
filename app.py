from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import subprocess
import translator
#import generator
import esprima
import escodegen
import re

app = Flask(__name__)
CORS(app)

"""
@app.route("/compile", methods=["POST"])
def compile_code():
    data = request.get_json()
    code = data["code"]

    # write the code to a .java file
    with open("code.java", "w") as f:
        f.write(code)

    # Compile the code using the Java Compiler (javac) command
    try:
        subprocess.run(["javac", "code.java"], capture_output=True, text=True)
        result = "Successfully Compiled"
    except subprocess.CalledProcessError as e:
        result = e.stderr
    # return the result of the compilation
    return jsonify({"result": result})

    """

@app.route('/', methods=['GET','POST'])
def generate():
    if request.method == 'POST':
        source_code = request.form['source_code']
        target_code = translator.translate(source_code)
        # target_code = "def add(x, y):\n    return x + y"
        return render_template('home.html', target_code=target_code)
    else:
        return render_template('home.html')


@app.route('/translate', methods=['POST'])
def translate():
    javascript_code = ""
    formatted_js_code = ""
    ast = ""
    if request.method == "POST":
        java_code = request.json['java_code']
        java_regex = r'"([^"\\]*(?:\\.[^"\\]*)*)"'
        java_string_literals = re.findall(java_regex, java_code)

        try:
            javascript_code = translator.translate(java_code)
        except:
            print("error occured in translation")
    
        try:
            ast = esprima.parseScript(javascript_code)
            formatted_js_code = str(escodegen.generate(ast))
            js_regex = r'\'([^"\\]*(?:\\.[^"\\]*)*)\''
            js_string_literals = [(match.group(1), match.start()) for match in re.finditer(js_regex, java_code)]
        except:
            print("error occured in parsing")
            formatted_js_code = javascript_code

    return {'javascript_code': javascript_code,
            'string_labels': java_string_literals,
            'ast': str(ast),
            'formatted_jscode' : formatted_js_code
            }

@app.route('/copy')
def copy():
    return jsonify({"message": "Code copied to clipboard"})

if __name__ == '__main__':
    app.run(debug=True)