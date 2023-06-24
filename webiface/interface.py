from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/work", methods=["POST"])
def work():
    return render_template("work.html")
