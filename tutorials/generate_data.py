from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from msmd.midi_parser import notes_to_onsets, FPS
from msmd.data_model.piece import Piece
from msmd.alignments import align_score_to_performance
import os
import csv
from PIL import Image
import ast
import cv2

DATA_ROOT_MSMD = '/data/mirlab/msmd/msmd_aug/msmd_aug'

def get_images(piece_name):
    """
    
    """
    piece = Piece(root=DATA_ROOT_MSMD, name=piece_name)

    if piece.available_scores:
        score = piece.load_score(piece.available_scores[0])
    
    mungos = score.load_mungos()
    mdict = {m.objid: m for m in mungos}
    mungos_per_page = score.load_mungos(by_page=True)
    images = score.load_images()
    
    # get only systems
    system_mungos = [c for c in mungos_per_page[0] if c.clsname == 'staff']
    system_mungos = sorted(system_mungos, key=lambda m: m.top)

    # get only noteheads
    notehead_mungos = [c for c in mungos_per_page[0] if c.clsname == 'notehead-full']

    return system_mungos, notehead_mungos, images


def generate_dict(img,notehead_mungos):
    """
    creates a dictionary with keys at y locations (strip center locations) whose values are the x coordinates of notes
    """
#     for sys_mungo in system_mungos:
#         t, l, b, r = sys_mungo.bounding_box
#         plt.plot([l, r], [t, t], c='g', linewidth=3, alpha=0.5)
#         plt.plot([l, r], [b, b], c='g', linewidth=3, alpha=0.5)
#         plt.plot([l, l], [t, b], c='g', linewidth=3, alpha=0.5)
#         plt.plot([r, r], [t, b], c='g', linewidth=3, alpha=0.5)

    y_list = []
    for n in notehead_mungos:
        y_list.append(n.middle[0])
        # plt.plot(n.middle[1], n.middle[0], "bo", alpha=0.5)
    uq_y = list(set(y_list))
    uq_y = sorted(uq_y)

    assoc_x = {}
    for y in uq_y:
        if y not in assoc_x.keys():
            assoc_x[y] = []
        for n in notehead_mungos:
            if n.middle[0] == y:
                assoc_x[y].append(n.middle[1]) # add associated x position on this strip
                
    return uq_y, assoc_x


def create_strips(uq_y, assoc_x, piece_name, page_num, system_mungos,img):
    with open('data.csv', 'a', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL, delimiter=':')
        for index in range(len(uq_y)):
            t, l, b, r = system_mungos[0].bounding_box
            t = uq_y[index] - 4
            b = uq_y[index] + 4
            top, left, bottom, right = t,l,b,r
        #     plt.figure("Strip", figsize=(10,20))
            strip = img[top:bottom, left:right]
#             print(type(strip))

#             plt.figure("strip") # , figsize=(16,20)
#             plt.imshow(strip, cmap="gray")

            curr_vals = assoc_x[uq_y[index]]
            for idx in range(len(curr_vals)):
                curr_vals[idx] -= l

            # generate vectors

            # save plot
            filename = piece_name+'_'+str(page_num)+'_'+str(index)
            #plt.axis('off')
            #plt.savefig(filename,bbox_inches = 'tight',pad_inches = 0)
            plt.imsave(filename, strip, format="png",cmap='Greys_r')
#             cv2.imwrite(filename, cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))    
            wr.writerow((filename, curr_vals)) # write filename and then current values associated w/ it
            

        
def main():
    piece_list = []
    DATA_ROOT_MSMD = '/data/mirlab/msmd/msmd_aug/msmd_aug'

    for root, dirs, filenames in os.walk(DATA_ROOT_MSMD):
        for directory in dirs:
            if "_" in directory and "ly" not in directory:
                piece_list.append(directory)

    for piece_name in piece_list[:100]:
        if "ly" in piece_name:
            continue
        print(piece_name)
        try:
            system_mungos, notehead_mungos, images = get_images(piece_name)
        except:
            continue
        img = images[0]
                
        uq_y, assoc_x = generate_dict(img, notehead_mungos)
        create_strips(uq_y, assoc_x, piece_name, 0,system_mungos,img)

main()

