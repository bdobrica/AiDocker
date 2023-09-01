from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/work", methods=["POST"])
def work():
    return render_template("work.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug="run")
