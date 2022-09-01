class Conv_ESN:
    def __init__(self, N_inputs, N_reservoir = 50, spectral_radius = 1, sparsity = 0, noise = 0.001, random_state = None):
        
        self.N_inputs = N_inputs
        self.N_reservoir = N_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.random_state = random_state

        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invalid seed: " + str(e))
        else:
            self.random_state_ = np.random.mtrand._rand

        self.initweights()
        
        
    def initweights(self):
        

        W = self.random_state_.rand(self.N_reservoir, self.N_reservoir) - 0.5
        W[self.random_state_.rand(*W.shape) < self.sparsity] = 0
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * (self.spectral_radius / radius)
        self.W_in = self.random_state_.rand(self.N_reservoir, self.N_inputs) * 2 - 1
        
        
    def _update(self, state, input_pattern):
        preactivation = (np.dot(self.W, state)+ np.dot(self.W_in, input_pattern))
        return (np.tanh(preactivation)+ self.noise * (self.random_state_.rand(self.N_reservoir) - 0.5))
    
    
    def fit_internal_state(self, inputs, inspect=False):
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))

        states = np.zeros((inputs.shape[0], self.N_reservoir))
        for n in range(1, inputs.shape[0]):
            states[n, :] = self._update(states[n - 1], inputs[n, :])
            
        extended_states = np.hstack((states, inputs))

        if inspect:
            from matplotlib import pyplot as plt
            plt.figure(figsize=(states.shape[0] * 0.0025, states.shape[1] * 0.01))
            plt.imshow(extended_states.T, aspect='auto',interpolation='nearest')
            plt.colorbar()
            print(states.shape)
        
        return states
