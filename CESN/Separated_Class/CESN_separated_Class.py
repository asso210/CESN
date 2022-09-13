class ESN:
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
        self.W_in = self.random_state_.rand(
                    self.N_reservoir, self.N_inputs) * 2 - 1
       
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
            plt.figure(
                figsize=(states.shape[0] * 0.0025, states.shape[1] * 0.01))
            plt.imshow(extended_states.T, aspect='auto',interpolation='nearest')
            plt.colorbar()
            print(states.shape)
        
        return states


N_inputs = 3
N_reservoir = 15
vec_internal_states_XYZ =[]
dim_temp_series = 1000
vec_output_Class = []


for i in range(4000):   
    #You have to replace np.random.rand with your value
    input_ESN_x = np.random.rand(dim_temp_series)
    input_ESN_y = np.random.rand(dim_temp_series)
    input_ESN_z = np.random.rand(dim_temp_series)
    
        esn = ESN(N_inputs = N_inputs,
                    N_reservoir = N_reservoir,
                    noise=0.1,
                    spectral_radius = 1.5,
                    random_state=40)

        input_ESN = np.array([input_ESN_x,input_ESN_y,input_ESN_z]).reshape(dim_temp_series,N_inputs)
        internal_states_XYZ = esn.fit_internal_state(input_ESN,inspect=False)        
        vec_internal_states_XYZ.append(internal_states_XYZ)
        
trainlen = 900
testlen = 300
shift_trainlen = 0
shift_testlen = 0

X_train = np.array(vec_internal_states_XYZ[0+shift_trainlen:trainlen+shift_trainlen])
Y_train = np.array(vec_output_Class[0+shift_trainlen:trainlen+shift_trainlen])
X_train = tf.stack(X_train)
Y_train = tf.stack(Y_train)
X_train = tf.expand_dims(X_train, axis=-1)

X_test = np.array(vec_internal_states_XYZ[trainlen+shift_testlen+shift_trainlen:trainlen+testlen+shift_trainlen+shift_testlen])
Y_test = np.array(vec_output_Class[trainlen+shift_testlen+shift_trainlen:trainlen+testlen+shift_testlen+shift_trainlen])
X_test = tf.expand_dims(X_test, axis=-1)


model=Sequential()
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(dim_temp_series,N_reservoir,1), kernel_regularizer=regularizers.l2(l=0.01)))
model.add(AveragePooling2D(pool_size=2))
model.add(BatchNormalization())
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu", kernel_regularizer=regularizers.l2(l=0.01)))
model.add(AveragePooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(80,activation="relu"))
model.add(Dense(20,activation="relu"))
model.add(Dense(2,activation="softmax"))

opt = SGD(lr=0.01)
opt_adam = 'adam'

binary_crossentropy = 'binary_crossentropy'

model.compile(loss=binary_crossentropy,optimizer=opt_adam,metrics=['accuracy'])
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30)

#### Validate the model predicting other value
start_len = 1200
stop_len = 2700
range_len = stop_len - start_len

X_test = np.array(vec_internal_states_XYZ[start_len:stop_len])
X_test = tf.expand_dims(X_test, axis=-1)

prediction_ = model.predict(X_test)
test_preds = np.where(prediction_ > 0.5, 1, 0)
