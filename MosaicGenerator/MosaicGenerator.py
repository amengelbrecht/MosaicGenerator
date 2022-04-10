import cv2
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import copy

class MosaicGenerator():
    def __init__(self):
        pass
    
    def extract_frames(self, skip_frames=1, image_size=256):
        count = 0
        
        for file in os.listdir('./source'):
            filename = os.fsdecode(file)
        
            cap = cv2.VideoCapture('./source/'+filename)
            if cap != None:
                print(f'Extracting {filename}...')
            else:
                print(f'Could not extract frames from {filename}')
                break
            success, image = cap.read()
    
            with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
                while success:
                    dim = int(min(image.shape[:-1])/2)
                    mid_w = int(image.shape[0]/2)
                    mid_h = int(image.shape[1]/2)
                    image = image[mid_w-dim:mid_w+dim, mid_h-dim:mid_h+dim,:]
                    image = cv2.resize(image, (image_size,image_size))
    
                    cv2.imwrite(f'./frames/{count}.jpg', image)
                    for _ in range(skip_frames):
                        pbar.update(1)
                        success, image = cap.read()
                    count += 1
    
    def process_images(self):
        n, b, g, r = [], [], [], []
        directory = os.fsencode('./frames/')
        
        for file in tqdm(os.listdir(directory)):
            filename = os.fsdecode(file)
            
            image = cv2.imread('./frames/' + filename)
            
            n.append(filename)
            b.append((image[:,:,0]).mean())
            g.append((image[:,:,1]).mean())
            r.append((image[:,:,2]).mean())
            
        self.df = pd.DataFrame(zip(n,b,g,r), columns=['n', 'b', 'g', 'r'])
        print('Done')
        
    def generate_image(self, target_filename, single_use=False):
        source_df = copy.deepcopy(self.df)
        target_image = cv2.imread(target_filename)
        width = target_image.shape[0]
        height = target_image.shape[1]
        
        collage = []
        
        print('Building image...')
        for w in tqdm(range(width)):
            row = []
            
            for h in range(height):
                diff = abs(source_df['b']-target_image[w,h,0]) + abs(source_df['g']-target_image[w,h,1]) + abs(source_df['r']-target_image[w,h,2])
                row.append(cv2.imread(f'./frames/{source_df.iloc[diff.idxmin()].n}'))
                
                if single_use:
                    source_df.iloc[diff.idxmin()] = ['0.jpg', float('inf'), float('inf'), float('inf')]
                    
            collage.append(np.hstack(row))
            
        output_image = np.vstack(collage)
        
        print('Writing image...', end='')
        print(cv2.imwrite('./output.png', output_image))
        
        return output_image