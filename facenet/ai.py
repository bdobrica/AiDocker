#!/usr/bin/env python3
import sys
import os
import re
import time
import datetime
import signal
import json
from urllib import request
from pathlib import Path
from daemon import Daemon
import cv2
import numpy as np
import dlib
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from keras.models import load_model
from sqlalchemy import create_engine,\
    MetaData, Table, Column, String, DateTime, Text
import base64
from io import BytesIO

class AIDaemon(Daemon):
    def ai(self, source_file, prepared_file, **metadata):
        pid = os.fork()
        if pid != 0:
            return

        try:
            im = cv2.imread(str(source_file))
            out_im = im.copy()

            face_detector = dlib.get_frontal_face_detector()

            MODEL_PATH = os.environ.get('MODEL_PATH', '/opt/app/facenet_keras.h5')
            facenet = load_model(MODEL_PATH)
            facenet_input = tuple(facenet.inputs[0].shape[1:])

            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            face_boxes = face_detector(gray)

            faces = None
            boxes = None
            for face_box in face_boxes:
                face = im[face_box.top():face_box.bottom(),
                    face_box.left():face_box.right()]
                face = cv2.resize(face, facenet_input[:2])
                face = face.reshape(1, *facenet_input)
                if not faces:
                    faces = face
                    boxes = [(face_box.top(), face_box.left(),
                        face_box.bottom(), face_box.right())]
                else:
                    faces = np.concatenate((faces, face), axis=0)
                    boxes.append((face_box.top(), face_box.left(),
                        face_box.bottom(), face_box.right()))

            if faces is not None:
                signatures = facenet.predict(faces)
                buffer = BytesIO()
                np.save(buffer, signatures)
                buffer.seek(0)
                base64_signatures = base64.b64encode(buffer.getvalue())

                DATABASE_PATH = os.environ.get('DATABASE_PATH', '/tmp/db')
                db_files = Path(
                    DATABASE_PATH,
                    metadata.get('partition', 'default')
                ).glob('*.npy')
                face_ids = []
                face_distances = None
                for db_file in db_files:
                    face_ids.append(db_file.stem)
                    face_cmp = np.load(db_file)

                    cos_distance = (np.matmul(signatures, face_cmp.T) /\
                        np.linalg.norm(signatures, axis = 1) /\
                        np.linalg.norm(face_cmp, axis = 1)).max(axis = 1)

                    if not face_distances:
                        face_distances = cos_distance.reshape(-1, 1)
                    else:
                        face_distances = np.concatenate((face_distances,\
                            cos_distance.reshape(-1, 1)), axis = 1)
                
                if face_distances is not None:
                    results = {
                        'signatures': base64_signatures,
                        'similarities': [ {
                            'id': face_ids[n],
                            'box': boxes[i],
                            'similarity': face_distances[i, n]
                        } for i, n in enumerate(face_distances.argmax(axis = 1)) ]
                    }
                else:
                    results = {
                        'signatures': base64_signatures,
                        'similarities': [ {
                            'id': 'unknown',
                            'box': box,
                            'similarity': 0
                        } for box in boxes ]
                    }

                for result in results:
                    label = '{face_id}:{similarity:.2f}%'.format(
                        face_id = result['id'],
                        similarity = 100 * result['similarity'])
                    box_top, box_left, box_bottom, box_right = result['box']
                    similarity = result['similarity']

                    color = [random.randint(0, 255) for _ in range(3)]
                    cv2.rectangle(
                        out_im,
                        (int(box_left), int(box_top)),
                        (int(box_right), int(box_bottom)),
                        color,
                        thickness=2)
                    
                    text_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
                    cv2.rectangle(
                        img_copy,
                        (int(box_left), int(box_top)),
                        (int(box_left + text_size[0]), int(box_top - text_size[1] - 3)),
                        color,
                        -1)
                    cv2.putText(
                        img_copy,
                        label,
                        (int(box_left), int(box_top - 2)),
                        0,
                        fontScale = 0.5,
                        color = (255, 255, 255),
                        thickness = 1,
                        lineType = cv2.LINE_AA)
            else:
                results = results = {
                    'signatures': '',
                    'similarities': []
                }
                
            json_file = prepared_file.with_suffix('.json')
            with json_file.open('w') as f:
                json.dump({'results':results}, f)
            
            if os.environ.get('API_DEBUG', False):
                cv2.imwrite(str(prepared_file), out_im)
        except Exception as e:
            pass
        
        source_file.unlink()
        sys.exit()

    def queue(self):
        STAGED_PATH = os.environ.get("STAGED_PATH", "/tmp/ai/staged")
        SOURCE_PATH = os.environ.get("SOURCE_PATH", "/tmp/ai/source")
        PREPARED_PATH = os.environ.get("PREPARED_PATH", "/tmp/ai/prepared")
        MAX_FORK = int(os.environ.get("MAX_FORK", 8))
        CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 4096))

        staged_files = sorted([ f for f in Path(STAGED_PATH).glob("*")\
            if f.is_file() and f.suffix != '.json'],
            key = lambda f : f.stat().st_mtime)
        source_files = [ f for f in Path(SOURCE_PATH).glob("*") if f.is_file() ]
        source_files_count = len(source_files)

        while source_files_count < MAX_FORK and staged_files:
            source_files_count += 1
            staged_file = staged_files.pop(0)

            meta_file = staged_file.with_suffix('.json')
            if meta_file.is_file():
                with meta_file.open('r') as fp:
                    try:
                        image_metadata = json.load(fp)
                    except:
                        image_metadata = {}
            image_metadata = {
                **{
                    'extension': staged_file.suffix,
                    'background': '',
                },
                **image_metadata
            }

            source_file = Path(SOURCE_PATH) / staged_file.name
            prepared_file = Path(PREPARED_PATH) /\
                (staged_file.stem + image_metadata['extension'])

            with staged_file.open('rb') as src_fp,\
                 source_file.open('wb') as dst_fp:
                while True:
                    chunk = src_fp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    dst_fp.write(chunk)

            staged_file.unlink()
            self.ai(source_file, prepared_file, **image_metadata)
    
    def check_db(self):
        DATABASE_PATH = os.environ.get('DATABASE_PATH', '/tmp/db')

        DATABASE_USER = os.environ.get('DATABASE_USER', 'root')
        DATABASE_PASSWORD = os.environ.get('DATABASE_PASSWORD', '')
        DATABASE_NAME = os.environ.get('DATABASE_NAME', '')
        DATABASE_TYPE = os.environ.get('DATABASE_TYPE', '')
        DATABASE_HOST = os.environ.get('DATABASE_HOST', '')
        DATABASE_TABLE = os.environ.get('DATABASE_TABLE', 'faces')

        latest_load_file = Path(DATABASE_PATH) / 'latest_load.txt'
        if latest_load_file.is_file():
           with latest_load_file.open('r') as fp:
               latest_load = float(fp.read())
        else:
            lastest_load = 0.0
        
        if time.time() - latest_load <\
            float(os.environ.get('DATABASE_CHECK_INTERVAL', 60)):
            return

        latest_date = datetime.datetime.utcfromtimestamp(seconds = latest_load)
    
        if not all(DATABASE_TYPE, DATABASE_HOST, DATABASE_NAME):
            return
        
        conn = self.db_engine.connect()

        if not hasattr(self, 'db_engine'):
            self.db_engine = create_engine(
                '{type}://{user}:{password}@{host}/{name}'.format(
                    type = DATABASE_TYPE,
                    user = DATABASE_USER,
                    password = DATABASE_PASSWORD,
                    host = DATABASE_HOST,
                    name = DATABASE_NAME
                )
            )
        
        if not self.db_engine.has_table(DATABASE_TABLE):
            metadata = MetaData(self.db_engine)
            faces_table = Table(
                DATABASE_TABLE,
                metadata,
                Column('name', String(255), primary_key=True),
                Column('signature', Text()),
                Column('updated_at', DateTime, default=func.now())
            )
            metadata.create_all()
        else:
            metadata = MetaData(self.db_engine)
            faces_table = Table(
                DATABASE_TABLE,
                metadata,
                autoload=True,
                autoload_with=self.db_engine
            )

        query = faces_table.select().where(faces_table.c.updated_at > latest_date)
        result = conn.execute(query)

        for row in result:
            row = dict(row)
            database_file = Path(DATABASE_PATH) / (row['name'] + '.npy')
            with database_file.open('wb') as fp:
                fp.write(base64.b64decode(row['signature']))
        
        with latest_load_file.open('w') as fp:
            fp.write(str(time.time()))


    def run(self):
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        while True:
            self.check_db()
            self.queue()
            time.sleep(1.0)
            
if __name__ == "__main__":
    CHROOT_PATH = os.environ.get("CHROOT_PATH", "/opt/app")
    PIDFILE_PATH = os.environ.get("PIDFILE_PATH", "/opt/app/run/ai.pid")

    AIDaemon(pidfile = PIDFILE_PATH, chroot = CHROOT_PATH).start()