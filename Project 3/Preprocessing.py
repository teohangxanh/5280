import numpy as np
from numpy import asarray
from pathlib import Path as p
from PIL import Image
import random
from typing import NamedTuple, List
from matplotlib import pyplot as plt
import os
import shutil


class Folder_data(NamedTuple):
    old_name: str
    new_name: str


class Var_data(NamedTuple):
    label: str
    img_arr: np.array


class Processor:
    '''
    This is a class that handles a zip file of multiple folders, which include labels and image titles
    '''

    def __init__(self, folder_path: str, train_frac: float = 0.8):
        self.folder_path = folder_path
        self.train_frac = train_frac  # Used to split into train and test sets
        self.train = []
        self.val = []
        self.test = []

    def to_folder(self, lst: list, sub_folder: str) -> List[Folder_data]:
        return [Folder_data(old_name=p(item[1]) / item[2], new_name=p(item[1]).parent / sub_folder / item[0]) for item
                in lst]

    def to_var(self, lst: list) -> List[Var_data]:
        return [Var_data(label=item[0].split('_', 1)[0],
                         img_arr=asarray(Image.open(p(item[1]) / item[2]))) for item in lst]

    def process_data(self) -> None:
        '''
           Save (image array, label) to self.train, self.val, and self.test
           Create folders of train, val, and test to the data directory
        '''
        # self.make_dir()
        for file in p(self.folder_path).iterdir():
            if file.is_dir():
                # Train = 0.8, val = 0.1, test = 0.1
                whole = [(f'{str(file.stem)}_{index}{f.suffix}', f.parent, f.name) for index, f in
                         enumerate(p(file).glob('*'), 1)
                         if f.suffix[1:] in {'jpeg', 'jpg', 'png'}]
                train = random.choices(whole, k=int(self.train_frac * len(whole)))
                whole = list(set(whole) - set(train))
                val = random.choices(whole, k=int(0.5 * len(whole)))
                test = list(set(whole) - set(val))
                # Save to folder
                # for f in self.to_folder(train, 'train'):
                #     shutil.copy(f.old_name, f.new_name)
                # for f in self.to_folder(val, 'val'):
                #     shutil.copy(f.old_name, f.new_name)
                # for f in self.to_folder(test, 'test'):
                #     shutil.copy(f.old_name, f.new_name)
                # Save to the instance
                self.train.extend(self.to_var(train))
                self.val.extend(self.to_var(val))
                self.test.extend(self.to_var(test))

    def make_dir(self):
        if os.path.exists(f'{self.folder_path}/train'):
            shutil.rmtree(f'{self.folder_path}/train')
        if os.path.exists(f'{self.folder_path}/val'):
            shutil.rmtree(f'{self.folder_path}/val')
        if os.path.exists(f'{self.folder_path}/test'):
            shutil.rmtree(f'{self.folder_path}/test')
        os.makedirs(f'{self.folder_path}/train')
        os.makedirs(f'{self.folder_path}/val')
        os.makedirs(f'{self.folder_path}/test')





folder_path = r'.\Hand extraction\target_box'
processor = Processor(folder_path, train_frac=0.8)
processor.process_data()
