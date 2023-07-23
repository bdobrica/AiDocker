#!/usr/bin/env python3
import json

from flask import Flask, Response

from .callbacks import __version__
from .callbacks.csv import get_csv, put_csv
from .callbacks.document import put_document
from .callbacks.image import get_image, put_image
from .callbacks.json import get_json
from .callbacks.text import put_text

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def get_root():
    return Response(
        json.dumps({"version": __version__}),
        status=200,
        mimetype="application/json",
    )


get_json = app.route("/get/json", methods=["POST"])(get_json)

put_image = app.route("/put/image", methods=["POST"])(put_image)
get_image = app.route("/get/image/<image_file>", methods=["GET"])(get_image)

put_text = app.route("/put/text", methods=["POST"])(put_text)

put_csv = app.route("/put/csv", methods=["POST"])(put_csv)
get_csv = app.route("/get/csv/<csv_file>", methods=["GET"])(get_csv)

put_document = app.route("/put/document", methods=["POST"])(put_document)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
