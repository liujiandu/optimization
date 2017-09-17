import imageio
imageio.plugins.ffmpeg.download()

from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy
import matplotlib.pyplot as plt
f = plt.figure(figsize=(8,8))
def make_frame_mpl(t):
	return mplfig_to_npiname(f)

animate = mpy.VideoClip(make_frame_mpl, duration=X_iter.shape[2]/40.)
#animate.write_gif('1.gif', fps=20)
