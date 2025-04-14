"""
To convert png files to video using imageio.v2 (memory-efficient version).
"""

# IMPORTs
import os

# IMPORTs sub
from glob import glob
import imageio.v2 as iio

# IMPORTs personal
from common import config, Decorators



class CreateVideo:
    """
    To create an MP4 video from png files.
    """

    @Decorators.running_time
    def __init__(self, filename: str, fps: int = 5) -> None:
        """
        Args:
            filename (str): the name of the video file to be created.
            fps (int, optional): the frames per second of the video. Defaults to 5.
        """

        # ATTRIBUTEs
        self.fps = fps
        self.filename = filename

        # RUN
        self.paths = self.paths_setup()
        self.mp4()

    def paths_setup(self) -> dict[str, str]:
        """
        To setup and format the paths for the images and video.

        Raises:
            FileNotFoundError: if the directory for images does not exist.

        Returns:
            dict[str, str]: the paths for the images and video.
        """

        # PATHs formatting
        paths = {
            'images': os.path.join(config.path.dir.data.result.projection, 'data_warped'),
            'video': config.path.dir.data.temp,
        }

        # CHECKs
        if not os.path.exists(paths['images']):
            raise FileNotFoundError(f"Directory {paths['images']} does not exist.")
        os.makedirs(paths['video'], exist_ok=True)
        return paths

    def mp4(self) -> None:
        """
        To create the mp4 video from the png files.
        """

        # PATHs images
        image_paths = sorted(glob(os.path.join(self.paths['images'], '*.png')))
        
        # VIDEO creation
        writer = iio.get_writer(
            os.path.join(self.paths['video'], self.filename),
            fps=self.fps
        )
        for path in image_paths: writer.append_data(iio.imread(path))
        writer.close()



if __name__ == '__main__':

    CreateVideo(filename='test.mp4', fps=10)
