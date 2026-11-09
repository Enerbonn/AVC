"""Synthetic visual regressions: movement must not erase content changes."""
import unittest
import cv2
import numpy as np

from src.motion import View, compare, annotate_motion, effective_analysis, circle_candidates, GrayCache
from src.selection import difference, thumbnail, apply_sensitivity


def page():
    image = np.full((270, 480), 255, np.uint8)
    for i in range(7):
        cv2.putText(image, f"ROW {i}   x + y = {i * 3}", (35, 55 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, 20, 1, cv2.LINE_AA)
    return image


def analysis(images):
    marks = []; group = 0
    for i, image in enumerate(images):
        d = difference(images[i - 1], image) if i else {"noise":False,"page_change":True,"changed_fraction":1,"removed_ink":0}
        if i and d['page_change']:group += 1
        marks.append({"frame_idx": i * 10, "kind":"burst_end", "included":True,
                      "visual_analysis":{**d,"camera":False,"page_group":group}})
    annotate_motion(marks, images, 10)
    effective=[]
    for m in marks:
        n=dict(m);n['visual_analysis']=effective_analysis(m);effective.append(n)
    return apply_sensitivity(effective,0)[0]


class MotionTests(unittest.TestCase):
    def test_evicted_analysis_proxies_are_lossless(self):
        cache=GrayCache(capacity=2)
        images=[page() for _ in range(8)]
        for i,image in enumerate(images):
            cv2.putText(image,str(i),(250,240),0,.5,0,1)
            cache.append(image)
        for i in [0,7,1,6,3]:
            self.assertTrue(np.array_equal(cache[i],images[i]))
            self.assertLessEqual(len(cache.cache),2)
    def test_tiny_new_stroke_is_a_candidate_not_fractional_pixel_noise(self):
        left=page();right=left.copy();cv2.line(right,(290,222),(303,222),0,2)
        result=difference(left,right)
        self.assertFalse(result['noise'])
    def test_moving_colored_pointer_does_not_become_pending_writing(self):
        pictures=[]
        for x in [75,185,285,385]:
            bgr=cv2.cvtColor(page(),cv2.COLOR_GRAY2BGR)
            cv2.circle(bgr,(x,230),12,(80,245,245),-1)
            cv2.line(bgr,(x,226),(x+3,234),(0,0,0),2)
            pictures.append(bgr)
        orb=cv2.ORB_create(nfeatures=900,fastThreshold=12)
        a,b=pictures[:2]
        metric=compare(View(cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),orb,circle_candidates(a)),
                       View(cv2.cvtColor(b,cv2.COLOR_BGR2GRAY),orb,circle_candidates(b)))
        self.assertEqual(metric['ink_added'],0)
        self.assertEqual(metric['ink_removed'],0)

    def test_panning_to_unseen_rows_preserves_both_views(self):
        canvas=np.full((620,480),255,np.uint8)
        for i in range(20):cv2.putText(canvas,f"UNIQUE ROW {i}",(30,30+i*28),0,.7,0,1)
        first=canvas[:270].copy();last=canvas[250:520].copy()
        marks=analysis([first,first,last,last])
        kept=[i for i,m in enumerate(marks) if m['included']]
        self.assertTrue(any(i<2 for i in kept))
        self.assertTrue(any(i>=2 for i in kept))

    def test_transient_occlusion_returns_to_a_clear_view(self):
        first=page();overlay=first.copy();cv2.rectangle(overlay,(60,65),(330,175),30,-1)
        marks=analysis([first,overlay,first])
        self.assertFalse(marks[1]['included'])
        demo,_=apply_sensitivity(marks,0,True,'demo')
        self.assertTrue(demo[1]['included'])

    def test_zoom_keeps_new_edge_content_eligible(self):
        first=page();zoom=cv2.warpAffine(first,np.float32([[1.1,0,-10],[0,1.1,-10]]),(480,270),borderValue=255)
        cv2.putText(zoom,"NEW HEADING",(5,18),0,.6,0,1)
        marks=analysis([first,zoom,zoom])
        self.assertTrue(any(m['included'] and m['frame_idx']>=10 for m in marks))

    def test_registered_translation_is_not_new_content(self):
        left=page();right=cv2.warpAffine(left,np.float32([[1,0,15],[0,1,8]]),(480,270),borderValue=255)
        orb=cv2.ORB_create(nfeatures=900,fastThreshold=12)
        result=compare(View(left,orb),View(right,orb))
        self.assertTrue(result['registered'])
        self.assertLess(result['added'],.06)
        self.assertLess(result['removed'],.06)
        marks=analysis([left,right,right])
        self.assertEqual(sum(m['included'] for m in marks),1)

    def test_new_writing_is_saved_before_erasure(self):
        first=page();written=first.copy()
        cv2.putText(written,"NEW = 19",(280,230),0,.7,0,2)
        blank=np.full(first.shape,255,np.uint8)
        marks=analysis([first,written,written,blank])
        self.assertTrue(any(m['included'] and m['frame_idx'] in [10,20] for m in marks))

    def test_local_repeating_animation_does_not_make_many_pages(self):
        images=[]
        for i in range(48):
            image=page()
            image[50:210,20:210]=0
            cv2.putText(image,str(i%4)*3,(25,145),0,1.4,255,3)
            images.append(image)
        marks=analysis(images)
        self.assertLessEqual(sum(m['included'] for m in marks),5)

    def test_a_small_symbol_addition_is_not_treated_as_an_exact_duplicate(self):
        left=page();right=left.copy();cv2.line(right,(290,222),(303,222),0,2)
        orb=cv2.ORB_create(nfeatures=900,fastThreshold=12)
        result=compare(View(left,orb),View(right,orb))
        self.assertGreater(result['new_pixels'],5)
        marks=analysis([left,right,left])
        self.assertTrue(marks[1]['included'], "Keep the new symbol before it is erased")
        demo,_=apply_sensitivity(marks,0,True,'demo')
        self.assertTrue(demo[1]['included'], "Demo mode must preserve before-loss content too")
