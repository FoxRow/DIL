import colorsys
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import webbrowser

import keras
import numpy as np
from keras import backend as K
from PIL import Image, ImageFont, ImageDraw
from yolo import yolo

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


class Program:
    PANO_TEMPLATE = """<!DOCTYPE HTML>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>A simple example</title>
        <link rel="stylesheet" href="https://cdn.pannellum.org/2.3/pannellum.css"/>
        <script type="text/javascript" src="https://cdn.pannellum.org/2.3/pannellum.js"></script>
        <style>
        #panorama {
            width: 600px;
            height: 400px;
        }
        </style>
    </head>
    <body>

    <div id="panorama"></div>
    <script>
    pannellum.viewer('panorama', {
        "type": "equirectangular",
        "panorama": "panorama.jpg",
        "autoLoad": true
    });
    </script>

    </body>
    </html>
    """

    def __init__(self, script):
        self.commands = self.parse(script)
        self.input_dir = tempfile.TemporaryDirectory()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.results_dir = tempfile.TemporaryDirectory()

    @staticmethod
    def parse(program):
        return [line.strip().split() for line in program.splitlines()]

    def load(self, new_images):
        self.input_dir.cleanup()
        self.input_dir = tempfile.TemporaryDirectory()
        self.temp_dir.cleanup()
        self.temp_dir = tempfile.TemporaryDirectory()

        for image in new_images:
            shutil.copy(image, self.input_dir.name)

    def get_images(self):
        return [os.path.join(self.input_dir.name, img) for img in os.listdir(self.input_dir.name)]

    def highlight(self, targets):
        session = K.get_session()

        with open('lib/classes.txt') as f:
            class_names = f.readlines()
        all_classes = [name.strip() for name in class_names]
        target_classes = [class_name.strip() for class_name in targets]

        anchors = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
        anchors = np.array(anchors).reshape(-1, 2)

        yolo_model = keras.models.load_model('lib/yolo.h5')
        yolo_outputs = yolo.yolo_head(yolo_model.output, anchors, len(all_classes))
        input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = yolo.yolo_eval(yolo_outputs, input_image_shape, score_threshold=.3, iou_threshold=.5)

        model_image_size = yolo_model.layers[0].input_shape[1:3]
        is_fixed_size = model_image_size != (None, None)

        hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        random.shuffle(colors)

        for image_path in self.get_images():
            image = Image.open(image_path)
            if is_fixed_size:
                resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
                image_data = np.array(resized_image, dtype='float32')
            else:
                new_image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
                resized_image = image.resize(new_image_size, Image.BICUBIC)
                image_data = np.array(resized_image, dtype='float32')

            image_data /= 255.
            image_data = np.expand_dims(image_data, 0)

            out_boxes, out_scores, out_classes = session.run([boxes, scores, classes],
                                                             feed_dict={yolo_model.input: image_data,
                                                                        input_image_shape: [image.size[1], image.size[0]],
                                                                        K.learning_phase(): 0})

            font = ImageFont.truetype(font='lib/FiraMono-Medium.otf',
                                      size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            thickness = (image.size[0] + image.size[1]) // 300

            for i, c in reversed(list(enumerate(out_classes))):
                predicted_class = class_names[c]
                if predicted_class not in target_classes:
                    continue

                box = out_boxes[i]
                score = out_scores[i]

                label = '{} {:.2f}'.format(predicted_class, score)

                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                for j in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                del draw

            image.save(image_path)
            shutil.copy(image_path, self.results_dir.name)

    def stack(self):
        raise NotImplementedError

    def stitch_pano(self):
        if platform.system() == 'Windows':
            pto_gen = 'C:/Program Files/Hugin/bin/pto_gen.exe'
            cpfind = 'C:/Program Files/Hugin/bin/cpfind.exe'
            cpclean = 'C:/Program Files/Hugin/bin/cpclean.exe'
            autooptimiser = 'C:/Program Files/Hugin/bin/autooptimiser.exe'
            pano_modify = 'C:/Program Files/Hugin/bin/pano_modify.exe'
            hugin_executor = 'C:/Program Files/Hugin/bin/hugin_executor.exe'
        else:
            pto_gen = 'pto_gen'
            cpfind = 'cpfind'
            cpclean = 'cpclean'
            autooptimiser = 'autooptimiser'
            pano_modify = 'pano_modify'
            hugin_executor = 'hugin_executor'

        if (not os.path.isfile(pto_gen) and
                not any(os.path.isfile(os.path.join(p, 'pto_gen')) for p in os.environ['PATH'].split(os.pathsep))):
            print('Hugin installation not found. Exiting.')
            return

        project_file = os.path.join(self.temp_dir.name, 'script.pto')

        pto_gen_command = [pto_gen, '-p', '2', '-o', project_file] + self.get_images()
        subprocess.run(pto_gen_command)

        cpfind_command = [cpfind, '--multirow', '-o', project_file, project_file]
        subprocess.run(cpfind_command)

        cpclean_command = [cpclean, '-o', project_file, project_file]
        subprocess.run(cpclean_command)

        autooptimiser_command = [autooptimiser, '-a', '-l', '-s', '-m', '-p', '-o', project_file, project_file]
        subprocess.run(autooptimiser_command)

        pano_modify_command = [pano_modify, '--ldr-file=JPG', '--ldr-compression=90', '--canvas=7000x3500',
                               '-o', project_file, project_file]
        subprocess.run(pano_modify_command)

        panorama_prefix = os.path.join(self.temp_dir.name, 'panorama')
        hugin_executor_command = [hugin_executor, '--stitching', '--prefix={}'.format(panorama_prefix), project_file]
        subprocess.run(hugin_executor_command)
        shutil.move(os.path.join(self.temp_dir.name, 'panorama.jpg'), self.results_dir.name)
        
        with open(os.path.join(self.results_dir.name, 'index.html'), 'w') as f:
            f.write(self.PANO_TEMPLATE)

    def show(self):
        pano_html = os.path.abspath(os.path.join(self.results_dir.name, 'index.html'))
        webbrowser.open(pano_html)

    def save(self):
        for result in os.listdir(self.results_dir.name):
            shutil.copy(os.path.join(self.results_dir.name, result), '.')

    def execute_command(self, command):
        if command[0] == 'load':
            self.load(command[1:])
        elif command[0] == 'highlight':
            self.highlight(command[1:])
        elif command[0] == 'stitch':
            self.stitch_pano()
        elif command[0] == 'show':
            self.show()
        elif command[0] == 'save':
            self.save()
        elif command[0] == '':
            pass
        elif command[0].startswith('#'):
            pass  # comment line
        else:
            print('unrecognized command: {}'.format(command[0]))

    def execute(self):
        for command in self.commands:
            self.execute_command(command)


if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        program = Program(f.read())
        program.execute()
