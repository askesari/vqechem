# Estimating ground state energy of Lithium Hydride (LiH) molecule

## Description
This project estimates the ground state energy of a LiH molecule in the following 3 scenarios:

- Ideal or noise-free AER simulator
- Noisy simulator
- Noisy simulator after applying M3 i.e. matrix-free measurement mitigation routine

The code runs in 2 phases:

**Phase I**: In this phase, the VQE algorithm is executed to identify the optimal parameter values and the related energy values e.g. extracted transformation energy, nuclear repulsion energy, etc. 

**Phase II**: In this phase, the ansatz circuit is updated to include post-rotation measurement circuits for all possible output states so that M3 routine can be applied. Subsequently, the updated circuits are transpiled and then executed for the 3 scenarios mentioned above. For the 3rd scenario i.e. noisy simulator with mitigation, correction is applied to the quasi-probabilities derived from the counts to compute the mitigated energy value.

## Pre-requisites
The pre-requisites for installing the package are:

### Python==3.8.13
It is advisable to create a new environment using either pip or conda to deploy the project. 
If using conda, the following command can be used where \<envname> needs to be replaced with the appropriate name during execution. 
    
    conda create --name <envname> python==3.8.13 

### Qiskit packages
- qiskit==0.38.0
    - qiskit-aer==0.11.0
    - qiskit-ibm-experiment==0.2.6
    - qiskit-ibmq-provider==0.19.2
    - qiskit-terra==0.21.2
- qiskit-nature==0.4.5
- qiskit_experiments==0.4.0

Following commands can be used to install the qiskit packages.

    pip install qiskit==0.38.0
    pip install qiskit-nature==0.4.5
    pip install qiskit_experiments==0.4.0

### PySCF library
- pyscf==2.1.1

Following command can be used to install the PySCF library.

    pip install pyscf==2.1.1

### YAML library
- PyYAML==6.0

Following command can be used to install the PyYAML library.

    pip install PyYAML==6.0

### MTHREE library
- mthree==1.1.0

Following command can be used to install the MTHREE library.

    pip install mthree==1.1.0

### Matplotlib library
- matplotlib==3.6.0
- pylatexenc==2.10

Following command can be used to install the Matplotlib library.

    pip install matplotlib==3.6.0
    pip install pylatexenc==2.10 

Alternatively, one can install all the necessary prerequisite packages and libraries by executing the following command. The requirements.txt file is provided in the repository.

    pip install -r requirements.txt

> Note: The qiskit account credentials must be stored on the machine on which the code is installed and executed. Refer to IBM Qiskit help to identify how qiskit account credentials can be stored locally

## Usage

### Basic Usage

Run the program *run_energy_estimator.py* at the command prompt using the command

    python3 run_energy_estimator.py

The above program will create the following output at the end of the execution.

    ****************************************************************
    The final output for LiH molecule for interatomic distance 1.4 Angstrom is =>
    Ideal simulator aer_simulator: Total energy value: -7.233931659579014 hartree
    Noisy simulator fake_vigo: Total energy value: -7.196506206040471 hartree
    Noisy simulator fake_vigo with M3 mitigation: Total energy value: -7.213376488825936 hartree
    Energy difference between noisy and ideal simulator: 0.03742545353854254 hartree
    Energy difference between noisy and ideal simulator post mitigation: 0.020555170753077334 hartree
    Percentage improvement in energy value calculation due to mitigation: 45.08
    ****************************************************************

Additionally, the ansatz circuit and one of the transpiled circuits (for both ideal and noisy simulators) will be created in the ./output folder. A sample Ansatz circuit is as follows.

![ansatz](https://github.com/askesari/vqechem/assets/13076705/a4c50acf-c618-4697-abb5-1e9516b2a583)

A sample transpiled circuit for noisy simulator is as follows.

![transCircuitNoisy_0](https://github.com/askesari/vqechem/assets/13076705/e2de9be5-86e8-4f65-8054-31d78d204035)

> Note: Executing the program will create an output and logs folders within the current directory.

### Advanced usage

There are several parameters that can be tweaked to execute different scenarios. The parameters are provided in the parameters.yaml file provided along with the package. The parameters are:

    # Allowed optimizer_label values: L-BFGS-B (default), COBYLA
    # Allowed mapper values: P (default, represents parity), JW (represents Jordan-Wigner)
    # Allowed entanglement_type values: full, linear (default)
    # Allowed noise_model_device values: "FakeVigo", "FakeAthens", "FakeManila"
    # Allowed enable_debug values: 0, 1

Additionally, following parameters can also be edited:

    interatomic_distance: 1.4 // this value is in Angstrom and can be edited to any appropriate value
    rotations: "ry" // this can be updated to "rz" or ["ry,rz"]
    depth: 2 // this indicates the number of layers of rotations parameter or depth of the circuit
    optimizer_maxiter: 50 // maximum number of iterations for the optimizer

> **Note 1**: Certain configurations may not work with the selected device e.g. Jordan-wigner transformation requires 6 qubits, whereas FakeVigo or FakeAthens have maximum 5 qubits. For this case, only FakeManila will work as it has 20 qubits.

After setting the appropriate parameters, execute the program *run_energy_estimator.py* as mentioned in the section **Basic usage**.

## Potential Bugs

1. Transpiling the circuits is an important step in the process before they are run to get the counts. Apparently, invoking transpiling for all circuits in one go throws a 'tkinter' exception which most probably is related to the mult-threading approach used by Qiskit's internal code:
    
        Exception ignored in: <function Image._del_ at 0x7f7ae4122c10>
        Traceback (most recent call last):
        File "/home/amit/miniconda3/envs/myqiskit/lib/python3.8/tkinter/_init.py", line 4017, in __del_
        self.tk.call('image', 'delete', self.name)
        RuntimeError: main thread is not in main loop
    So, to work around this issue, we transpile the circuits one at a time. While this may impact the performance slightly, it avoids the exception in almost all cases. Nonetheless, there might be a corner case, wherein the exception resurfaces.

## References

1. [Simulating molecules using the VQE algorithm on Qiskit](https://arxiv.org/pdf/2201.04216.pdf)
2. [Variational Quantum EigenSolver FOR "LiH" molecule](https://www.linkedin.com/pulse/variational-quantum-eigensolver-lih-molecule-krishanu-krishanu-podder)
3. [Scalable Mitigation of Measurement Errors on Quantum Computers](https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.2.040326)

## Author(s)

Amit Shashikant Kesari

## License
[Apache2.0](https://opensource.org/licenses/Apache-2.0)
