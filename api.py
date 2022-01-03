#!/usr/bin/env python3
import os
from io import BytesIO
from flask import Flask, request, Response, send_file
from pathlib import Path
from hashlib import md5, sha256
import base64
import json
import time

app = Flask(__name__)

@app.route('/put/image', methods=['POST'])
def put_image():
    STAGED_PATH = os.environ.get("STAGED_PATH", "/tmp/ai/staged")
    SOURCE_PATH = os.environ.get("SOURCE_PATH", "/tmp/ai/source")
    PREPARED_PATH = os.environ.get("PREPARED_PATH", "/tmp/ai/prepared")

    image_file = request.files.get('image')
    if not image_file:
        return Response(json.dumps({'error': 'missing image'}), status = 400, mimetype='application/json')

    image_type = image_file.mimetype
    image_data = image_file.read()

    image_hash = ({
        'MD5': md5,
        'SHA256': sha256
    }.get(os.environ.get('API_IMAGE_HASHER', 'SHA256').upper()) or sha256)()
    image_hash.update(image_data)
    image_token = image_hash.hexdigest()

    if image_type == 'image/png':
        staged_file = Path(STAGED_PATH) / (image_token + '.png')
        source_file = Path(SOURCE_PATH) / (image_token + '.png')
        prepared_file = Path(PREPARED_PATH) / (image_token + '.png')
        json_file = Path(PREPARED_PATH) / (image_token + '.json')
    elif image_type == 'image/jpeg':
        staged_file = Path(STAGED_PATH) / (image_token + '.jpg')
        source_file = Path(SOURCE_PATH) / (image_token + '.jpg')
        prepared_file = Path(PREPARED_PATH) / (image_token + '.png')
        json_file = Path(PREPARED_PATH) / (image_token + '.json')
    else:
        return Response(json.dumps({'error': 'unknown image format'}), status = 400, mimetype='application/json')

    if not (staged_file.is_file() or source_file.is_file() or prepared_file.is_file()):
        with staged_file.open('wb') as fp:
            fp.write(image_data)

    return Response(json.dumps({'token': image_token, 'mime': image_type}), status = 200, mimetype='application/json')

@app.route('/get/image', methods=['POST'])
def get_image():
    image_token = request.json.get('token')
    image_type = request.json.get('mime')

    STAGED_PATH = os.environ.get("STAGED_PATH", "/tmp/ai/staged")
    SOURCE_PATH = os.environ.get("SOURCE_PATH", "/tmp/ai/source")
    PREPARED_PATH = os.environ.get("PREPARED_PATH", "/tmp/ai/prepared")
    FILE_LIFETIME = os.environ.get('API_CLEANER_FILE_LIFETIME', '1800.0')

    lifetime = float(FILE_LIFETIME)

    if image_type == 'image/png':
        staged_file = Path(STAGED_PATH) / (image_token + '.png')
        source_file = Path(SOURCE_PATH) / (image_token + '.png')
        prepared_file = Path(PREPARED_PATH) / (image_token + '.png')
        json_file = Path(PREPARED_PATH) / (image_token + '.json')
    elif image_type == 'image/jpeg':
        staged_file = Path(STAGED_PATH) / (image_token + '.jpg')
        source_file = Path(SOURCE_PATH) / (image_token + '.jpg')
        prepared_file = Path(PREPARED_PATH) / (image_token + '.png')
        json_file = Path(PREPARED_PATH) / (image_token + '.json')
    else:
        return Response(json.dumps({'error': 'unknown image format'}), status = 400, mimetype='application/json')

    token_expired = staged_file.is_file() and (staged_file.stat().st_mtime + lifetime < time.time())

    if prepared_file.is_file() and not token_expired:
        if staged_file.is_file():
            staged_file.unlink()
        if source_file.is_file():
            source_file.unlink()
        if json_file.is_file():
            json_file.unlink()
        prepared_file.unlink()

        return send_file(BytesIO(image_data), mimetype = 'image/png', as_attachment = True, download_name = prepared_file.name)

    if staged_file.is_file() and not token_expired:
        return Response(json.dumps({'wait': 'true', 'status': 'not queued'}), status = 200, mimetype = 'application/json')

    if source_file.is_file() and not token_expired:
        return Response(json.dumps({'wait': 'true', 'status': 'processing'}), status = 200, mimetype = 'application/json')

    if token_expired:
        if staged_file.is_file():
            staged_file.unlink()
        if source_file.is_file():
            source_file.unlink()
        if json_file.is_file():
            json_file.unlink()
        if prepared_file.is_file():
            prepared_file.unlink()

        return Response(json.dumps({'error': 'token expired'}), status = 400, mimetype = 'application/json')

    return Response(json.dumps({'error': 'unknown token'}), status = 400, mimetype = 'application/json')

@app.route('/get/json', methods=['POST'])
def get_json():
    image_token = request.json.get('token')
    image_type = request.json.get('mime')

    STAGED_PATH = os.environ.get("STAGED_PATH", "/tmp/ai/staged")
    SOURCE_PATH = os.environ.get("SOURCE_PATH", "/tmp/ai/source")
    PREPARED_PATH = os.environ.get("PREPARED_PATH", "/tmp/ai/prepared")
    FILE_LIFETIME = os.environ.get('API_CLEANER_FILE_LIFETIME', '1800.0')

    lifetime = float(FILE_LIFETIME)

    if image_type == 'image/png':
        staged_file = Path(STAGED_PATH) / (image_token + '.png')
        source_file = Path(SOURCE_PATH) / (image_token + '.png')
        prepared_file = Path(PREPARED_PATH) / (image_token + '.png')
        json_file = Path(PREPARED_PATH) / (image_token + '.json')
    elif image_type == 'image/jpeg':
        staged_file = Path(STAGED_PATH) / (image_token + '.jpg')
        source_file = Path(SOURCE_PATH) / (image_token + '.jpg')
        prepared_file = Path(PREPARED_PATH) / (image_token + '.png')
        json_file = Path(PREPARED_PATH) / (image_token + '.json')
    else:
        return Response(json.dumps({'error': 'unknown image format'}), status = 400, mimetype='application/json')
    
    token_expired = staged_file.is_file() and (staged_file.stat().st_mtime + lifetime < time.time())

    if json_file.is_file() and not token_expired:
        with json_file.open('rb') as fp:
            json_data = json.load(fp)
        if staged_file.is_file():
            staged_file.unlink()
        if source_file.is_file():
            source_file.unlink()
        if prepared_file.is_file():
            prepared_file.unlink()
        json_file.unlink()

        if not isinstance(json_data, dict):
            return Response(json.dumps({'error': 'invalid model output'}), status = 400, mimetype='application/json')
        
        json_data.update({'token': image_token, 'mime': image_type, 'status': 'success'})

        return Response(json.dumps(json_data), status = 200, mimetype='application/json')

    if staged_file.is_file() and not token_expired:
        return Response(json.dumps({'wait': 'true', 'status': 'not queued'}), status = 200, mimetype = 'application/json')

    if source_file.is_file() and not token_expired:
        return Response(json.dumps({'wait': 'true', 'status': 'processing'}), status = 200, mimetype = 'application/json')

    if token_expired:
        if staged_file.is_file():
            staged_file.unlink()
        if source_file.is_file():
            source_file.unlink()
        if json_file.is_file():
            json_file.unlink()
        if prepared_file.is_file():
            prepared_file.unlink()

        return Response(json.dumps({'error': 'token expired'}), status = 400, mimetype = 'application/json')

    return Response(json.dumps({'error': 'unknown token'}), status = 400, mimetype = 'application/json')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)