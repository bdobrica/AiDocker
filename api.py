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

def file_paths(image_token, image_extension = None):
    STAGED_PATH = Path(os.environ.get("STAGED_PATH", "/tmp/ai/staged"))
    SOURCE_PATH = Path(os.environ.get("SOURCE_PATH", "/tmp/ai/source"))
    PREPARED_PATH = Path(os.environ.get("PREPARED_PATH", "/tmp/ai/prepared"))

    meta_file = STAGED_PATH / (image_token + '.json')
    if image_extension is None and meta_file.is_file():
        with meta_file.open() as fp:
            try:
                image_meta = json.load(fp)
            except:
                image_meta = {}
        image_extension = image_meta.get('extension', None)

    json_file = PREPARED_PATH / (image_token + '.json')
    if image_extension is None:
        staged_file = STAGED_PATH / (image_token + image_extension)
        source_file = SOURCE_PATH / (image_token + image_extension)
        prepared_file = PREPARED_PATH / (image_token + image_extension)
    else:
        staged_file = STAGED_PATH.glob(image_token + '.*')
        source_file = SOURCE_PATH.glob(image_token + '.*')
        prepared_file = PREPARED_PATH.glob(image_token + '.*')
    
    return {
        'meta_file': meta_file,
        'json_file': json_file,
        'staged_file': staged_file,
        'source_file': source_file,
        'prepared_file': prepared_file
    }

def clean_files(image_token):
    paths = file_paths(image_token)
    for path in paths.values():
        if isinstance(path, Path):
            path.unlink()
        else:
            for path_ in path:
                path_.unlink()

def clean_expired(image_token = None):
    STAGED_PATH = Path(os.environ.get("STAGED_PATH", "/tmp/ai/staged"))
    lifetime = float(os.environ.get('API_CLEANER_FILE_LIFETIME', '1800.0'))

    for meta_file in STAGED_PATH.glob('*.json'):
        if meta_file.stem == image_token:
            continue
        with meta_file.open() as fp:
            try:
                image_meta = json.load(fp)
            except:
                image_meta = {}
        if image_meta.get('upload_time', 0) + lifetime < time.time():
            clean_files(meta_file.stem)

@app.route('/put/image', methods=['POST'])
def put_image():
    image_file = request.files.get('image')
    if not image_file:
        return Response(
            json.dumps({'error': 'missing image'}),
            status = 400,
            mimetype='application/json')
    
    image_background = request.form.get('background', '')
    image_background = image_background.strip('#')

    image_type = image_file.mimetype
    image_data = image_file.read()

    image_hash = ({
        'MD5': md5,
        'SHA256': sha256
    }.get(os.environ.get('API_IMAGE_HASHER', 'SHA256').upper()) or sha256)()
    image_hash.update(image_data)
    image_token = image_hash.hexdigest()

    with open('mimetypes.json', 'r') as fp:
        image_extension = json.load(fp).get(image_type, '.jpg')

    image_metadata = {
        'token': image_token,
        'background': image_background,
        'type': image_type,
        'extension': image_extension,
        'upload_time': time.time(),
        'processed': 'false'
    }

    paths = file_paths(image_token, image_extension)

    meta_file = paths['meta_file']

    if not meta_file.is_file():
        with meta_file.open('w') as fp:
            json.dump(image_metadata, fp)

    return Response(
        json.dumps({'token': image_token}),
        status = 200,
        mimetype='application/json')

@app.route('/get/image/<image_token>')
def get_image(image_token):
    clean_expired(image_token)

    paths = file_paths(image_token)
    prepared_file = paths['prepared_file']

    if not prepared_file.is_file():
        return Response(
            'image not found',
            status = 404)

    with prepared_file.open('rb') as fp:
        image_data = fp.read()
    if not image_data:
        return Response(
            'image not found',
            status = 404)

    meta_file = paths['meta_file']
    if meta_file.is_file():
        with meta_file.open() as fp:
            try:
                image_metadata = json.load(fp)
            except:
                image_metadata = {}

    image_type = image_metadata.get('type', 'image/jpeg')
    image_extension = image_metadata.get('extension', '.jpg')
    
    clean_files(image_token, image_extension)

    return send_file(
        BytesIO(image_data),
        mimetype = image_type,
        as_attachment = True,
        download_name = image_token + image_extension
    )

@app.route('/get/json', methods=['POST'])
def get_json():
    lifetime = float(os.environ.get('API_CLEANER_FILE_LIFETIME', '1800.0'))

    image_token = request.json.get('token')
    paths = file_paths(image_token)

    meta_file = paths['meta_file']
    if not meta_file.is_file():
        clean_files(image_token)
        return Response(
            json.dumps({'error': 'unknown token'}),
            status = 400,
            mimetype = 'application/json')
    
    try:
        image_metadata = json.load(meta_file.open('r'))
    except:
        clean_files(image_token)
        return Response(
            json.dumps({'error': 'corrupted image metadata'}),
            status = 400,
            mimetype = 'application/json')
    
    if image_metadata.get('upload_time', 0) + lifetime < time.time():
        clean_files(image_token)
        return Response(
            json.dumps({'error': 'token expired'}),
            status = 400,
            mimetype = 'application/json')
    
    if image_metadata.get('extension') is None:
        clean_files(image_token)
        return Response(
            json.dumps({'error': 'invalid image extension'}),
            status = 400,
            mimetype = 'application/json')
    
    json_file = paths['json_file']
    prepared_file = paths['prepared_file']

    if json_file.is_file() and not prepared_file.is_file():
        with json_file.open('r') as fp:
            try:
                json_data = json.load(fp)
            except:
                json_data = {}
        
        if not json_data:
            clean_files(image_token)
            return Response(
                json.dumps({'error': 'invalid model output'},
                status = 400,
                mimetype = 'application/json'))
        
        json_data.update({
            'token': image_token,
            'status': 'success'
        })

        return Response(
            json.dumps(json_data),
            status = 200,
            mimetype='application/json')
    
    staged_file = paths['staged_file']
    source_file = paths['source_file']

    if prepared_file.is_file() and not json_file.is_file():
        if staged_file.is_file():
            staged_file.unlink()
        if source_file.is_file():
            source_file.unlink()
        
        return Response(
            json.dumps({'url': '/get/image/{image_token}'.format(
                image_token = image_token
            )}),
            status = 200,
            mimetype = 'application/json'
        )
    
    if source_file.is_file():
        return Response(
            json.dumps({'wait': 'true', 'status': 'processing'}),
            status = 200,
            mimetype = 'application/json')

    if staged_file.is_file():
        return Response(
            json.dumps({'wait': 'true', 'status': 'not queued'}),
            status = 200,
            mimetype = 'application/json')

    return Response(
        json.dumps({'error': 'unknown token'}),
        status = 400,
        mimetype = 'application/json')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
