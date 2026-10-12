
import os
import subprocess as sp

base = "/ceph/behrens/meg/3x3_field_magnitude_bandit/derivatives/cohort-01/sub-03_id-MY_71_R/"

for dirname in os.listdir(base):
    #print(dirname)
    name = str(dirname)
    if name[:5] == "ses-2":
        num = name[4:7]
        #print(num)

        #os.makedirs(f"./data/ses-{num}")
        
        for run in os.listdir(f"{base}{dirname}/behav"):
            runname = str(run)
            if run[:3] == "run":
                #print(run)
                
                command = ["cp", f"{base}{dirname}/behav/{runname}/imaging_frames_behavior.pkl", f"./data/ses-{num}/{runname}.pkl"]
        
                print(command)
                sp.run(command)



