import unittest
import cv2
import numpy as np
from src.ink import colored_ink
from src.motion import annotate_motion, circle_candidates
from src.selection import difference, thumbnail, apply_sensitivity

def select(images, sensitivity=0):
    grays=[thumbnail(i) for i in images];marks=[];group=0
    for n,g in enumerate(grays):
        d=difference(grays[n-1],g) if n else dict(noise=False,page_change=True,changed_fraction=1,removed_ink=0)
        if n and d['page_change']:group+=1
        marks.append(dict(frame_idx=n*10,kind='burst_end',included=True,visual_analysis={**d,'camera':False,'page_group':group}))
    annotate_motion(marks,grays,10,circles=[circle_candidates(i) for i in images],colors=[colored_ink(i) for i in images])
    return apply_sensitivity(marks,sensitivity)[0]

def sheet():
    im=np.full((270,480,3),255,np.uint8)
    for n in range(6):cv2.putText(im,f'LINE {n} A + B = {n}',(30,45+n*25),0,.65,(0,0,0),1,cv2.LINE_AA)
    return im

class InkTests(unittest.TestCase):
    def test_fading_emphasis_does_not_force_completed_summary(self):
        base=sheet();sequence=[base]
        for _ in range(4):
            for c in [(0,0,180),(0,0,180),(100,100,210),(180,180,235),(255,255,255)]:
                im=base.copy();cv2.line(im,(35,51),(150,51),c,2);sequence.append(im)
        marks=select(sequence)
        self.assertLessEqual(sum(m['included'] for m in marks),2)
        detailed=select(sequence,50)
        self.assertGreater(sum(m['included'] for m in detailed),sum(m['included'] for m in marks))
    def test_colored_new_symbol_in_blank_space_survives_erasure(self):
        base=sheet();written=base.copy();cv2.line(written,(290,230),(303,230),(0,0,180),2)
        marks=select([base,written,base])
        self.assertTrue(marks[1]['included'])
    def test_real_writing_partial_fade_is_coalesced(self):
        base=sheet();sequence=[base]
        for c in [0,0,80,130,180,210,235,255]:
            im=base.copy();cv2.putText(im,'NEW=12',(290,230),0,.6,(c,c,c),2);sequence.append(im)
        marks=select(sequence)
        forced=[m for m in marks if m['visual_analysis']['motion']['summary'].get('force_keep')]
        self.assertLessEqual(len(forced),2)
        self.assertTrue(any(m['included'] and m['frame_idx'] in [10,20,30] for m in marks))
    def test_black_tiny_symbol_with_brief_appearance_is_protected(self):
        base=sheet();written=base.copy();cv2.line(written,(290,230),(303,230),(0,0,0),2)
        marks=select([base,written,base])
        self.assertTrue(marks[1]['included'])
    def test_registered_dragging_preserves_unseen_rows(self):
        canvas=np.full((620,480,3),255,np.uint8)
        for n in range(20):cv2.putText(canvas,f'UNIQUE ROW {n}',(30,30+n*28),0,.7,(0,0,0),1)
        marks=select([canvas[:270],canvas[:270],canvas[250:520],canvas[250:520]])
        self.assertTrue(any(m['included'] and m['frame_idx']<20 for m in marks))
        self.assertTrue(any(m['included'] and m['frame_idx']>=20 for m in marks))
