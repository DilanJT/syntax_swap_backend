from flask import Flask, request, jsonify, render_template
import subprocess
import translator
#import generator
app = Flask(__name__)

# @app.route('/')
# def hello_world():
#     return "Welcome to SyntaxSwap!"

# This is the route that will be called when the user clicks the "Compile" button
# This is will compile the code that the user has written in the editor and return the result of the compilation

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


@app.route('/copy')
def copy():
    return jsonify({"message": "Code copied to clipboard"})

if __name__ == '__main__':
    app.run(debug=True)