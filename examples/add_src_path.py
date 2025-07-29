import sys
import os
dir_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dir_src = os.path.join(dir_parent, 'src')
sys.path.append(dir_src)
