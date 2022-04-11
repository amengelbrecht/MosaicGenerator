import cv2
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import copy

def extract_frames(skip_frames=1, image_size=256, source_dir='./source'):
    count = 0
    
    # Check that source directory exists
    if not os.path.exists(source_dir):
        print(f'{source_dir} does not exist.')
        return
    # Make frames directory if it doesn't exist
    if not os.path.exists('./frames'):
        os.makedirs('./frames')

    for file in os.listdir(source_dir):
        filename = os.fsdecode(file)

        cap = cv2.VideoCapture(source_dir + '/' +filename)
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

def process_images():
    n, b, g, r = [], [], [], []
    directory = os.fsencode('./frames')
    
    for file in tqdm(os.listdir(directory)):
        filename = os.fsdecode(file)

        image = cv2.imread('./frames/' + filename)

        n.append(filename)
        b.append((image[:,:,0]).mean())
        g.append((image[:,:,1]).mean())
        r.append((image[:,:,2]).mean())

    return pd.DataFrame(zip(n,b,g,r), columns=['n', 'b', 'g', 'r'])

def generate_image(target_filename, df=None, single_use=False):
    # Generate df if not included in fn call
    if df == None:
        print('No DataFrame passed. Generating now...')
        df = process_images()
    
    source_df = copy.deepcopy(df)
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
    if cv2.imwrite('./output.png', output_image):
        print('done.')
    else:
        print('failed.')

    return output_image