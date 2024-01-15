"""
Used to separate the mp.4 images from Sir Auchere's presentation.
"""

# Imports
import os
import cv2
import datetime


class DataFinder:

    def __init__(self):
        

        # Functions
        self.Paths()

    def Paths(self):

        main_path = '/home/avoyeux/Desktop/avoyeux'
        self.paths = {'Main': main_path,
                      'Screenshots': os.path.join(main_path, 'Screenshots3'),
                      'Stereo': os.path.join(main_path, 'STEREO', 'ratio'),
                      'MP4': os.path.join(main_path, '..', 'backup_2023/code_stuff/Rainbow/media/media8.mp4'),
                      'Saving': os.path.join(main_path, 'MP4_saves')
        }
        os.makedirs(self.paths['Saving'], exist_ok=True)


class Cutting_mp4(DataFinder):
    """
    Cutting the mp4 images.
    """

    def __init__(self):
        super().__init__()

        # Functions
        self.Cutting()

    def Cutting(self):
        cap = cv2.VideoCapture(self.paths['MP4'])


        start_date = datetime.datetime(2012, 7, 23, 0, 0)

        rank = -1
        while True:
            success, frame = cap.read()

            if not success:
                break
            rank += 1 

            new_date = start_date + datetime.timedelta(minutes=10*rank)
            frame_name = f'Frame_{new_date.month:02d}m{new_date.day:02d}d_{new_date.hour:02d}h{new_date.minute:02d}.png'
            frame_path = os.path.join(self.paths['Saving'], frame_name)

            cv2.imwrite(frame_path, frame)
        
        cap.release()
        print('Done.')


if __name__=='__main__':
    Cutting_mp4()