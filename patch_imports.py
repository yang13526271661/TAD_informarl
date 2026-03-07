import sys
file_path = "/home/yangxiaodi/yangxiaodi_space/TAD-informarl/InforMARL-main/multiagent/TAD_rand_2t1a1d.py"
with open(file_path, "r") as f:
    text = f.read()

text = text.replace("from onpolicy.envs.mpe.core import", "from multiagent.TAD_core import")
text = text.replace("from onpolicy.envs.mpe.scenario import", "from multiagent.scenario import")

# Remove the sys.path append block
target_sys = "import sys\nsys.path.append('/data/goufandi_space/Projects/deception_TAD_marl/onpolicy/envs/mpe/scenarios/')\nfrom util import *"
text = text.replace(target_sys, "from multiagent.TAD_util import *")

with open(file_path, "w") as f:
    f.write(text)

