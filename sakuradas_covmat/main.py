from Params import *
import sys
sys.path.append("utils/")
from cov_utils import network_covmat, network_covmat_sakbin

if __name__ == "__main__":
    
    #network_covmat()
    network_covmat_sakbin()