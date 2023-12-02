"""
The purpose of this module is to estimate the ground state energy of LiH molecule module
Author(s): Amit S. Kesari
"""

from qiskit import QuantumCircuit, QuantumRegister, Aer, IBMQ
from qiskit.compiler import transpile
import yaml, os
import numpy as np
from GroundStateEnergyEstimation import EnergyEstimator, plot_intermediate_results_graph, \
                                        variational_eigen_solver, exact_eigen_solver, \
                                        plot_energy_graph
from QcExecution import QcExecution as qce
from qiskit.utils import QuantumInstance, algorithm_globals
from M3Mitigation import M3MitigationDerived as m3
#import scipy.optimize as opt
from logconfig import get_logger

## define some global variables
#outputfilepath = '/home/amit/IITM/Tayur Prize 2022/output'
curr_dir_path = os.path.dirname(os.path.realpath(__file__))
outputfilepath = curr_dir_path + "/output"

if not os.path.exists(outputfilepath):
    os.makedirs(outputfilepath)

## initialize logger
log = get_logger(__name__)

seed = 170
algorithm_globals.random_seed = seed

def is_folder_path_exists(folderpath):
    """
    Check if the folder path location exists and return True, if yes, otherwise False
    """
    ## initialize to False
    folder_exists = False

    try:
        if folderpath is not None:
            #v_file_dir = filename_with_path.rpartition("/")[0]
            try:
                if not os.path.exists(folderpath):
                    raise NameError("Folder path does not exist: " + folderpath)
            except NameError as ne:
                log.exception(ne, stack_info=True)
                raise
            else:
                folder_exists = True
        else:
            raise NameError("Folder path not passed as input: " + folderpath)          
    except NameError as ne1:
        log.exception(ne1, stack_info=True)
        raise
    
    return(folder_exists) 

def opstr_to_meas_circ(op_str):
    """
    Description: Takes a list of operator strings and makes circuit with the correct post-rotations 
    for measurements.

    Parameters:
        op_str (list): List of strings representing the operators needed for measurements.

    Returns:
        list: List of circuits for measurement post-rotations
    """
    num_qubits = len(op_str[0])
    circs = []
    for op in op_str:
        qc = QuantumCircuit(num_qubits)
        for idx, item in enumerate(op):
            if item == 'X':
                qc.h(num_qubits-idx-1)
            elif item == 'Y':
                qc.sdg(num_qubits-idx-1)
                qc.h(num_qubits-idx-1)
        circs.append(qc)
    return circs

def cost_func(params, *args):
    # Get the required arguments
    backend = args[0]
    trans_circs = args[1]
    mit = args[2]
    coeffs = args[3]
    meas_strings = args[4]

    # Attach parameters to the transpiled circuit variables
    bound_circs = [circ.bind_parameters(params) for circ in trans_circs]
    # Submit the job and get the resultant counts back
    #counts = qi.execute(circuits=bound_circs, had_transpiled=True).get_counts()
    counts = backend.run(bound_circs, shots=4096).result().get_counts()
    
    # Apply mitigation to get quasi-probabilities if requested
    # else get the quasi-probabilities based only on original counts
    if mit is not None:
        quasis = mit.apply_mitigation(counts=counts)
        log.debug(f"Applied Mitigation: {quasis[0]}")
    else:
        quasis = m3.no_mitigation(counts=counts)
        log.debug(f"No Mitigation: {quasis[0]}")

    # Evaluate the coefficients times each expectation value obtained from quasi-probabilities and sum.
    energy = np.sum(coeffs*quasis.expval(meas_strings))
    return energy

# start of main function
def main():
    log.info("=============================================")
    log.info(f"Start of program ...")
    log.info(f"Checking if output path exists ...")
    outputpath_exists = is_folder_path_exists(outputfilepath)

    log.info(f"Loading parameter file ...")
    ## load the parameter.yaml file
    skip_lines=10
    try:
        with open("parameters.yaml", 'r') as param_stream:
            for i in range(skip_lines):
                _ = param_stream.readline()
            parameters = yaml.safe_load(param_stream)
    except FileNotFoundError as fnf:
        raise
    finally:
        param_stream.close()
    
    log.info(f"paramaters: {parameters}")
    log.info(f"Parameter file read successfully.")

    ### Set the parameters for LiH molecule
    remove_orbitals = parameters['lih_molecule']['remove_orbitals'] #for LiH
    z2symmetry_reduction = parameters['lih_molecule']['z2symmetry_reduction'] # for LiH

    # Set the parameters for initial state and build the ansatz
    rotations = parameters['lih_molecule']['rotations']
    entanglement = parameters['lih_molecule']['entanglement']
    entanglement_type = parameters['lih_molecule']['entanglement_type']
    depth = parameters['lih_molecule']['depth']

    optimizer_label = parameters['lih_molecule']['optimizer_label']
    optimizer_maxiter = parameters['lih_molecule']['optimizer_maxiter']
    distance = parameters['lih_molecule']['interatomic_distance']

    mapper = parameters['lih_molecule']['mapper']
    #is_simulator = parameters['lih_molecule']['is_simulator']
    noise_model_device = parameters['lih_molecule']['noise_model_device']
    #add_exact_solution = bool(parameters['lih_molecule']['add_exact_solution'])
    enable_debug = parameters['lih_molecule']['enable_debug']

    is_simulator = 1 ## we only run in simulator mode
    add_exact_solution = bool(0) ## we don't run the NumPyEigenSolver
    is_debug = enable_debug

    log.info(f"Parameters assigned successfully as follows.")
    log.info(f"Inter-atomic distance of LiH molecule: {distance} Angstrom")
    log.info(f"Molecule optimization: orbital reduction: {remove_orbitals}, z2symmetry: {z2symmetry_reduction}")
    log.info(f"Electronic to qubit converter/mapper code: {mapper}")
    log.info(f"Ansatz creation parameters: rotations: {rotations}, entanglement: {entanglement}, entanglement type: {entanglement_type}")
    log.info(f"Noisy simulator device: {noise_model_device}")
    log.info(f"Optimizer label for VQE: {optimizer_label}")
    log.info(f"Running program in debug mode (0 - No, 1 - Yes)?: {enable_debug}")

    # load the qiskit account for execution
    log.info(f"Loading Qiskit account. Please wait as this may take a while ...")
    qce.load_account()
    log.info(f"Qiskit account loaded successfully.")

    energies = dict()

    """
    ***************************************************************
    Phase I: In this first Phase, we run VQE on a noise-free ideal simulator to identify the 
    optimal values of the parameters used within the Ansatz. Additionally, we also identify extra
    energy terms such as nuclear repulsio energy, energy due to applied transformations, etc. which
    when included in the eigen value of the reduced hamiltonian will give us the total energy value.
    ***************************************************************
    """

    ### H2 molecule details
    #geometry = [["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, 0.735]]]

    ### LiH molecule details
    geometry = [["Li", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, distance]]]
    
    ## initialize energy estimator object
    ee = EnergyEstimator(geometry, z2symmetry_reduction, remove_orbitals, mapper=mapper, 
                        debug=is_debug) 
    qubit_op  = ee.get_hamiltonian_op(debug=is_debug)
    
    num_particles = ee.electronic_structure_problem.num_particles
    num_spin_orbitals = ee.electronic_structure_problem.num_spin_orbitals
    num_qubits = qubit_op.num_qubits

    #print(qubit_op)
    log.info(f"Number of particles:  {num_particles}, Number of spin orbitals: {num_spin_orbitals}, Number of qubits: {num_qubits}")

    init_state = ee.set_initial_state(debug=is_debug)
    if  outputpath_exists == True:
       qcex = qce(init_state,ee.electronic_structure_problem.num_spin_orbitals)
       qcex.draw_circuit(outputfilepath + '/' + 'initial_state.png')
    
    ansatz = ee.build_ansatz(num_qubits, init_state, rotations, entanglement, entanglement_type, depth, 
                                debug=is_debug)
    if outputpath_exists == True:
        #qcex = qce(ansatz,ee.electronic_structure_problem.num_spin_orbitals)
        #qcex.draw_circuit(outputfilepath + '/' + 'ansatz.png')
        ansatz.decompose().draw(output='latex',filename=outputfilepath + '/' + 'ansatz.png')
        log.info(f"Ansatz printed. Check for file ansatz.png in the output directory.")


    #execute VQE algorithm for the ideal simulator
    qcex = qce(QuantumCircuit(), num_qubits)
    ## ignore noise model and coupling map as they are not applicable for ideal simulator
    ideal_backend, _, _ = qcex.get_backend(is_simulator=bool(is_simulator), 
                                                            simulator_type='AER',
                                                            noise_model_device=None
                                                        )

    ## Also, get the noisy backend to be used later in Phase II of computation 
    ## Ignore the "noise model" and "coupling map" values returned by the method
    noisy_backend, _, _ = qcex.get_backend(is_simulator=bool(is_simulator), 
                                                            simulator_type='AER',
                                                            noise_model_device=noise_model_device
                                                        )
                                                        
    myoptimizer = EnergyEstimator.get_optimizer(optimizer_label=optimizer_label, 
                                                maxiter = optimizer_maxiter,
                                                debug=is_debug)
    ## First execute VQE on an ideal/noise free simulator
    #if is_simulator == True:
    #    qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed,
    #                 coupling_map=coupling_map, noise_model=noise_model)
    #else:
    """
    qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)
    if noise_model is not None:
        noisy_qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed,
                     coupling_map=coupling_map, noise_model=noise_model)
    else:
        noisy_qi = qi
    """

    log.info(f"Starting phase I of computation ...")
    # execute VQE on ideal/noise-free simulator
    log.info(f"Solving VQE. Please wait as this may take a while ...... ")
    execute_vqe = variational_eigen_solver(ansatz, optimizer=myoptimizer, 
                                            quantum_instance=ideal_backend)
    result = execute_vqe.compute_minimum_eigenvalue(qubit_op)
    es_result = ee.electronic_structure_problem.interpret(result)
    log.info(f"Hurray! VQE computation is complete. We now have the optimal parameter values.")

    if is_debug == True:
        log.debug(es_result)

    ## Get Phase 1 output of VQE
    energies['TE_VQE'] = es_result.total_energies.real[0]
    energies['CE_VQE'] = es_result.eigenenergies.real[0]
    energies['NRE'] = np.real(es_result.nuclear_repulsion_energy)
    energies['ETE'] = np.real(es_result.extracted_transformer_energy)
    optimal_parameters = result.optimal_point
    if is_debug == True:
        log.debug(f"Optimal Parameters: {optimal_parameters}")
    
    # Also invoke the exact eigen solver if parameter set to True
    if add_exact_solution == True:
        exact_eigen_value = exact_eigen_solver(qubit_op)
        energies['TE_NUMPY'] = exact_eigen_value + energies['NRE'] + energies['ETE']

    if add_exact_solution == True:
        log.info(f"Distance: {distance}, Eigenvalue: {result.eigenvalue.real}, Exact Eigenvalue: {exact_eigen_value}, Total: {es_result.total_energies.real[0]}, Nuclear replusion:  {es_result.nuclear_repulsion_energy}")
    else:
        log.info(f"Distance: {distance}, Eigenvalue: {result.eigenvalue.real}, Total: {es_result.total_energies.real[0]}, Nuclear replusion: {es_result.nuclear_repulsion_energy}")

    intermediate_results_plot_loc = outputfilepath + '/' + 'intermediate_results_plot.png'
    plt1 = plot_intermediate_results_graph(debug=is_debug)
    plt1.savefig(intermediate_results_plot_loc)
    plt1.close()

    if is_debug == True:
        log.debug(f"Energy values after VQE computation on ideal simulator ==> {energies}")
  
    """
    *******************************************************
    Phase II: Here, we leverage the output of Phase I to update the Ansatz with the 
              post-rotation and measurement gates, set the parameters to values around the 
              optimal values captured in phase I and compute the minimal value
              using SciPy package for 3 cases:
              1. Ideal/noisefree simulator
              2. Noisy simulator
              3. Noisy simulator with error mitigation
    *******************************************************
    """

    log.info(f"Starting phase II of computation ...")
    ## Step 1: Get the operator list (pauli strings) from the qubit opertor
    qubit_op_list_with_coeffs = ee.get_qubit_op_strings_and_coeffs(debug=is_debug)
    op_list = [item[0] for item in qubit_op_list_with_coeffs]
    coeffs = np.array([np.real(item[1]) for item in qubit_op_list_with_coeffs], dtype=float)
    if is_debug == True:
        log.debug(f"Operator list: {op_list}")
        log.debug(f"Coefficients: {coeffs}")

    ## Step 2: Get post-rotation and measurement circuits for the operator list
    try:
        meas_circs = opstr_to_meas_circ(op_list)
    except Exception as e:
        log.exception(e, stack_info=True)
        raise
    log.info(f"Post-rotation and measurement circuits identified for X and Y operators.")
    
    ## Step 2(b): Replace X and Y operator strings with Z
    try:
        meas_strings = [string.replace('X', 'Z').replace('Y', 'Z') for string in op_list]
    except Exception as e:
        log.exception(e, stack_info=True)
        raise
    log.info(f"X and Y operator strings now replaced with Z string.")

    ## Step 3: Compose full circuits by appending measurement circuits to the Ansatz
    try:
        full_circs = [ansatz.compose(meas_circs[kk]).measure_all(inplace=False) for kk in range(len(meas_circs))]
    except Exception as e:
        log.exception(e, stack_info=True)
        raise
    log.info(f"Post-rotation and measurement circuits appended to ansatz to get full circuits.")
    log.info(f"Number of full measurement cicuits: {len(full_circs)}")

    ## Step 4: Generate transpiled circuits for ideal simulator
    trans_circs = []
    try:
        trans_circs = [transpile(x, ideal_backend) for x in full_circs]
        #trans_circs = transpile(full_circs, ideal_backend)
    except Exception as e:
        log.exception(e, stack_info=True)
        raise
    log.info(f"Transpiled circuits generated for the backend: {ideal_backend}")
    if outputpath_exists == True:
        trans_circs[0].decompose().draw(output='latex',filename=outputfilepath + '/' + 'transCircuitIdeal_0.png')
        log.info(f"One of the transpiled circuits printed. Check for file transCircuitIdeal_0.png in the output directory.")

    ## Step 5: Resue the optimal parameter values for further computation
    params = np.array(optimal_parameters)
    #params += 0.05*(np.random.random(params.shape[0])-0.5)
    log.info(f"Leveraging optimal parameter values for further computation.")
    if is_debug == True:
        log.debug(f"Parameter values: {params}")

    ## Step 7: Call the cost_function
    ##
    ## backend = args[0]
    ## trans_circs = args[1]
    ## mit = args[2] = None
    ## coeffs = args[3]
    ## meas_strings = args[4]
    log.info(f"Identifying optimal energy value for ideal simulator. Please wait as this may take a while ...... ")

    result_ideal = cost_func(params, ideal_backend, trans_circs, None, coeffs, meas_strings)
    log.info(f"Expectation value for ideal simulator {ideal_backend}: {result_ideal}")

    ideal_total_energy = result_ideal + energies['NRE'] + energies['ETE']
    energies['CE_IDEAL'] = result_ideal
    energies['TE_IDEAL'] = ideal_total_energy

    ## Step 8: Generate the transpiled circuits for the noisy backend
    """
    IMP: Apparently, invoking transpiling for all circuits in one go throws a 'tkinter' exception:
            Exception ignored in: <function Image._del_ at 0x7f7ae4122c10>
            Traceback (most recent call last):
            File "/home/amit/miniconda3/envs/myqiskit/lib/python3.8/tkinter/_init.py", line 4017, in __del_
            self.tk.call('image', 'delete', self.name)
            RuntimeError: main thread is not in main loop
        Workaround: Execute transpilation one circuit at a time to avoid qiskit's multithreaded approach
    """
    trans_circs = []
    try:
        trans_circs = [transpile(x, noisy_backend) for x in full_circs]
        #trans_circs = transpile(full_circs, noisy_backend)
    except Exception as e:
        log.exception(e, stack_info=True)
        raise
    log.info(f"Transpiled circuits generated for the backend: {noisy_backend}")
    if outputpath_exists == True:
        trans_circs[0].draw(output='latex',filename=outputfilepath + '/' + 'transCircuitNoisy_0.png')
        log.info(f"One of the transpiled circuits printed. Check for file transCircuitNoisy_0.png in the output directory.")

    ## Step 9: Call the cost function for noisy simulator, but no mitigation
    ##
    ## backend = args[0]
    ## trans_circs = args[1]
    ## mit = args[2] = None
    ## coeffs = args[3]
    ## meas_strings = args[4]
    log.info(f"Identifying optimal energy value for noisy simulator. Please wait as this may take a while ...... ")

    result_noisy = cost_func(params, noisy_backend, trans_circs, None, coeffs, meas_strings)
    log.info(f"Expectation value for noisy simulator {noisy_backend}: {result_noisy}")

    noisy_total_energy = result_noisy + energies['NRE'] + energies['ETE']
    energies['CE_NOISY'] = result_noisy
    energies['TE_NOISY'] = noisy_total_energy

    ## Step 10: setup mitigation object for noisy simulator with calibration values
    mit = m3(backend=noisy_backend, num_qubits=num_qubits)
    log.info(f"M3 mitigation object set.")

    ## Step 11: Call the cost function for noisy simulator post mitigation
    ##
    ## backend = args[0]
    ## trans_circs = args[1]
    ## m3 = args[2]
    ## coeffs = args[3]
    ## meas_strings = args[4]
    log.info(f"Identifying optimal energy value after applying mitigation over nosiy simulator. Please wait as this may take a while ...... ")

    result_noisy_mit = cost_func(params, noisy_backend, trans_circs, mit, coeffs, meas_strings)
    log.info(f"Expectation value for noisy simulator {noisy_backend} after mitigation: {result_noisy_mit}")

    noisy_mit_total_energy = result_noisy_mit + energies['NRE'] + energies['ETE']
    energies['CE_MIT_NOISY'] = result_noisy_mit
    energies['TE_MIT_NOISY'] = noisy_mit_total_energy

    # Compute the delta between the total energies for the 3 cases
    delta_ideal_noisy = energies['TE_NOISY'] - energies['TE_IDEAL']
    delta_ideal_noisy_mit = energies['TE_MIT_NOISY'] - energies['TE_IDEAL']
    percent_change = round(-100  * (delta_ideal_noisy_mit - delta_ideal_noisy) / delta_ideal_noisy, 2)

    if is_debug == True:
        log.debug(f"Final energy values are: {energies}")
    log.info(f"End of program.")

    print(f"****************************************************************")
    print(f"The final output for LiH molecule for interatomic distance {distance} Angstrom is =>")
    if add_exact_solution == True:
        print(f"**Only for reference** Ideal simulator {ideal_backend}: Total energy using NumpySolver: {energies['TE_NUMPY']} hartree")
    print(f"Ideal simulator {ideal_backend}: Total energy value: {energies['TE_IDEAL']} hartree")
    print(f"Noisy simulator {noisy_backend}: Total energy value: {energies['TE_NOISY']} hartree") 
    print(f"Noisy simulator {noisy_backend} with M3 mitigation: Total energy value: {energies['TE_MIT_NOISY']} hartree") 
    print(f"Energy difference between noisy and ideal simulator: {delta_ideal_noisy} hartree")
    print(f"Energy difference between noisy and ideal simulator post mitigation: {delta_ideal_noisy_mit} hartree")
    print(f"Percentage improvement in energy value calculation due to mitigation: {percent_change}")
    print(f"****************************************************************")

if __name__ == '__main__':
    main()
