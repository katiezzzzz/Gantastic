from moviepy.editor import *
import os

PATH = os.path.dirname(os.path.realpath(__file__))
img = []
for img_path in ['snow1', 'snow2', 'snow3', 'city1', 'city2', 'city3']:
    file = PATH + '/../earth_screenshots/{}.jpg'.format(img_path)
    img.append(file)

clips = [ImageClip(m).set_duration(2)
      for m in img]

concat_clip = concatenate_videoclips(clips, method="compose")
concat_clip.write_videofile("test.mp4", fps=24)