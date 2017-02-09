"""
Runs MotionTracking.py for all studies and all feature detectors.
"""
import subprocess

studies = ['andre_nostamp','andre_stamp1','andre_stamp2',
           'yidi_nostamp','yidi_stamp1','yidi_stamp2']
           
fDs = ['sift','surf','orb','brisk']

command = "python MotionTracking.py {} {} {}"

for i in [1,2]:
    for study in studies:
        for detector in fDs:
            subprocess.call(command.format(study,detector,i))