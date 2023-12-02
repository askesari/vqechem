"""
This module contains the module supports mitigation.
Author(s): Amit S. Kesari
"""
from mthree import M3Mitigation
from mthree.utils import counts_to_vector, vector_to_quasiprobs
from mthree.classes import QuasiCollection
import numpy as np
from logconfig import get_logger

## initialize logger
log = get_logger(__name__)

class M3MitigationDerived(M3Mitigation):
    """
    This class enables in setting up the mitigation calibration counts.
    """

    def __init__(self, backend, num_qubits=None):
        """
        Description: initialize the base class and setup calibration counts
        """
        super().__init__(system=backend)
        ## Identify the maximum number of qubits supported by the backend to ensure
        ## that the calibration is not done beyond that; otherwise it will results in
        ## an exception
        max_num_qubits = backend.configuration().n_qubits
        if num_qubits is not None:
            try:
                if num_qubits > max_num_qubits:
                    raise Exception(f"Number of qubits {num_qubits} greater than allowed number of qubits {max_num_qubits} for the device {backend}")
                else:
                    self.num_qubits = num_qubits
            except Exception as e:
                log.exception(e, stack_info=True)
                raise

        self.qubit_list = np.asarray(range(self.num_qubits))
        log.info(f"Qubit List for calibration: {self.qubit_list}")

        # Setup calibration data
        self.cals_from_system(self.qubit_list)
    
    def apply_mitigation(self, counts):
        """
        Description: Apply mitigation to the counts and return a Quasi-collection i.e. collection
        of quasi-probabilities
        Input Arguments:
        counts (dict, list): Input counts dict or list of dicts.
        qubits (array_like): Qubits over which to correct calibration data. Default is all.
        """
        log.info(f"Inside apply_mitigation method")
        qubits = self.qubit_list

        return(self.apply_correction(counts, qubits=qubits))
    
    @classmethod
    def no_mitigation(cls, counts):
        """
        Description: Return counts as Quasi-collection i.e. collection of quasi-probabilities
        without applying mitigation
        Input Arguments:
        counts (dict, list): Input counts dict or list of dicts.
        """
        log.info(f"Inside no_mitigation method")
        quasi_out = []
        for idx, cnts in enumerate(counts):
            mycnts = dict(cnts)
            sorted_cnts = dict(sorted(mycnts.items()))
            vec = counts_to_vector(sorted_cnts)
            qbdist = vector_to_quasiprobs(vec, sorted_cnts)
            #print(qbdist)
            quasi_out.append(qbdist)
        
        log.info(f"Non-mitigated counts converted to Quasi-collection.")
        return(QuasiCollection(quasi_out))
