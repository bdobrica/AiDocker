import os
from typing import Union

from flask import Response, redirect, render_template, request, session


def login_page() -> Union[Response, str]:
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username == os.getenv("WEB_UI_USERNAME", "admin") and password == os.getenv("WEB_UI_PASSWORD", "admin"):
            session["username"] = username
            return redirect(request.referrer)
        else:
            session["username"] = ""
            del session["username"]
            return render_template("login.html", username=username, error="Invalid username or password")
    return render_template("login.html", page_title="Login", username="", error="")


def logout_page() -> Response:
    session["username"] = ""
    del session["username"]
    return redirect(request.referrer)
