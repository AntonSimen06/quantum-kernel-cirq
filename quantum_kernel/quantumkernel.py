import cirq_google
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class QuantumKernel:

    """

    Atributes:

        train_data -> Pandas dataframe.
        test_data -> Pandas dataframe.
        num_classes -> Integer with number of classes in the data.
        mapping -> String with the type of feature mapping (can be 'X', 'Z' or 'ZZ').
        entanglement -> String with level of entanglement in ansatz (can be 'full', 'linear', or 'circular').
        num_layers -> Integer that is the number of layers in ansatz.
        simulator -> Simulator() class from cirq.

    """

    def __init__(self, train_data, test_data, num_classes, mapping, entanglement, num_layers, simulator = cirq.Simulator()):

        self.train_data = train_data
        self.test_data = test_data
        self.num_classes = num_classes
        self.mapping = mapping
        self.entanglement = entanglement
        self.num_layers = num_layers
        self.simulator = simulator

    def feature_map(self, x):

        """
        Create quantum-enhanced feature map, f, whose task is mapping data 
        to a higher-order (Hilbert) space, being K(x,y) = <f(x), f(y)>.

        Args:
            x       -> individual feature vector from normalized data.
            mapping -> String with the type of mapping. Can be 'X', 'Z' or 'ZZ'.

        Output: 
            qc -> Circuit object.
        """

        qc = cirq.Circuit()
        qubits = cirq.LineQubit(self.train_data.shape[1])

        if self.mapping == 'X':

            for qubit, feature in zip(qubits, x):
                qc.append(cirq.rx(feature).on(qubit))

        return qc

    def ansatz(self, params):

        """
        Tunable quantum circuit, U(w), so that U(w)|i> = |s(w)>.

        Args:
            params -> Array with tunable parameters.
        Output:
            qc -> Tunable Circuit object.        
        """

        qc = cirq.Circuit()
        num_qubits = self.train_data.shape[1]
        qubits = cirq.LineQubit(num_qubits)

        if self.entanglement == 'full':

            for layer in range(self.num_layers):
                for param, qubit in enumerate(qubits):
                    qc.append(rz(params[param + layer*num_qubits]).on(qubit))
                    qc.append(ry(params[param + layer*num_qubits]).on(qubit))

                if layer == self.num_layers - 1:
                    break

                for control, target in zip(range(num_qubits), range(num_qubits)):
                    if control < target:
                        qc.append(cirq.CNOT(control, target))

        return qc

    def cost_function(self, params):
        
        """
        Cost function of the model. The MSE was used but the idea is testing others (LogLoss, Empirical Risk, etc).
        Here we choose a parity function as being a Pauli observable (P = ZIII...III for binary classification) 
        whose expected value is measured and the cost function is evaluated for each feature vector through

                            L(x,y) = (< ZIII...III > - y)^2

        Args:
            params -> an array of tunable parameters
        Outputs:
            loss -> A float as being the mean squared error (MSE)

        """

        losses = []
        for index in range(self.train_data.shape[0]):
            
            qc = cirq.Circuit()
            qc.append(self.feature_map(list(self.train_data[ : -1 , index])))
            qc.append(self.ansatz(params))

            # Simulate the circuit.
            result = simulator.simulate(qc)
            final_state = result.final_state_vector

            if self.num_classes == 2:

                # expected value of ZIII...III operator
                exp_value=0
                for eigstate in final_state[ : int(len(final_state)/2) ]:
                    exp_value += abs(eigstate)**2
                for eigstate in final_state[ int(len(final_state)/2) : ]:
                    exp_value -= abs(eigstate)**2            

                # Loss function
                label = self.train_data[-1, index]
                losses.append((exp_value - label)**2)         

        return sum(losses)/self.train_data.shape[0]

    def train(self):

        """
        Training the quantum circuit using SciPy optimizers.
        Outputs:
            res -> A dictionary with results from minimize() function from SciPy.
            convergence -> An array with loss function for each epoch.
        """

        print(" Training quantum circuit... \n\n")

        iteration=1
        convergence = []
        def callback(variational_parameters):
            global iteration
            convergence.append(self.cost_function(variational_parameters))
            print("Iteration: ", iteration, " \t Loss: ",  self.cost_function(variational_parameters))
            iteration += 1

        res = scipy.optimize.minimize(self.cost_function, x0=np.random.uniform(0, np.pi, 
                                        self.num_layers*self.train_data.shape[1]), 
                                        method = 'SLSQP', callback=callback,
                                        options={'maxiter': 200, 'ftol': 1e-06, 'iprint': 1, 'disp': True, 
                                        'eps': 1.4901161193847656e-08, 'finite_diff_rel_step': None})

        return res, convergence

    def train_and_test():

        """
        Testing data from test_data dataframe.
        """
        # training to obtain the optmized parameters of the ansatz
        optimal_params = self.train()[0]['x']

        trues = []

        for index in range(self.test_data.shape[0]):
            
            qc = cirq.Circuit()
            qc.append(self.feature_map(list(self.test_data[ : -1 , index])))
            qc.append(self.ansatz(optimal_params))

            # Simulate the circuit.
            result = simulator.simulate(qc)
            final_state = result.final_state_vector

            if self.num_classes == 2:

                # expected value of ZIII...III operator
                exp_value=0
                for eigstate in final_state[ : int(len(final_state)/2) ]:
                    exp_value += abs(eigstate)**2
                for eigstate in final_state[ int(len(final_state)/2) : ]:
                    exp_value -= abs(eigstate)**2                         

                # Validation
                label = self.test_data[-1, index]

                if label == -1 and exp_value < 0:
                    trues.append(0)
                elif label == 1 and exp_value > 0:
                    trues.append(0)

        print(f"Accuracy: {100*len(trues)/self.test_data.shape[0]} %")


# for testing, fill with arguments and uncomment bellow
"""

args = (

    train_data =
    test_data =
    num_classes =
    mapping =
    entanglement =
    num_layers =
    simulator =
    

)

if you want to get convergence and optimization results, use:
results, convergence = QuantumKernel(args).train()

if you want to train as well as testing, use:
QuantumKernel(args).train_and_test()

"""