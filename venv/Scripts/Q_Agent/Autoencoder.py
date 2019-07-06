from collections import deque
from keras.models import Sequential,Model
from keras.layers import Dense, Flatten, Dropout, Conv2D , LSTM, MaxPooling2D, GlobalAveragePooling2D,Input,UpSampling2D
from keras.optimizers import Adam
from keras.regularizers import l1,l2
from keras import backend as K
import numpy as np
import random
import math
import retro
import keyboard
import time
import matplotlib.pyplot as plt
LEVELS = ['BustAMove.1pplay.Level10','BustAMove.1pplay.Level20','BustAMove.1pplay.Level30',
           'BustAMove.1pplay.Level40','BustAMove.1pplay.Level50','BustAMove.1pplay.Level60',
           'BustAMove.1pplay.Level70','BustAMove.1pplay.Level80','BustAMove.1pplay.Level90',
           'BustAMove.1pplay.Level1',]
CHALLENGE= ['BustAMove.Challengeplay1','BustAMove.Challengeplay2',
           'BustAMove.Challengeplay3','BustAMove.Challengeplay4']
ESTADOS = CHALLENGE
class Q_Agent:
    def __init__(self,state_size,action_size):
        self.state_size = state_size
        self.action_size = action_size

        #NUMERO DE ACCIONES A "RECORDAR"
        self.memory = []
        #RAZON DE DESCUENTO
        self.gammaComp = 0.05
        self.gammaIncrease = 0.99995
        self.minGammaComp = 0.05
        #RAZON DE EXPLORACION
        self.epsilon = 0.05
        #EPSILON MINIMO
        self.min_epsilon = 0.05
        #DESCUENTO DEL EPSILON
        self.decay_epsilon = 0.9999         #DESCUENTO DEL DESCUENTO
        #self.decay_decay = 0.99999
        #RAZON DE APRENDIZAJE
        self.learning_rate = 0.0005
        #MODELO CREADO
        self.model =self._build_model()
        self.target_model = self._build_model()

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1 + K.square(error)) - 1, axis=-1)

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        print("guarda3")
        self.model.save_weights(name)

    def _build_model(self):
        input_img = Input(shape=self.state_size)
        #model.add(Conv2D(15,input_shape=self.state_size,kernel_size=(5,5),strides=(2,2),activation='relu'))
        #model.add(Flatten())
        #model.add(Dense(200, activation='relu'))
        #model.add(Dense(30, activation='relu'))
        #model.add(Dense(self.action_size, activation='linear'))
        #aqui hay aprox 64512 valores
        x = Conv2D(16, (4, 4), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (4, 4), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (4, 4), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        # aqui hay aprox 2700 valores

        x = Conv2D(8, (4, 4), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (4, 4), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (4, 4), activation='relu',padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, (4, 4), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        autoencoder.summary()
        return autoencoder

    def remember(self,state):
        self.memory.append((state))

    def act(self, state):
        accion =random.randrange(self.action_size)
        return accion


    def toBinary(self,action):
        if action==3:
            return np.zeros(12)
        accion = [0,6,7][action]
        return np.concatenate((np.zeros(accion), np.array([1]), np.zeros(11 - accion)))

    def replay(self, batch_size):

        minibatch = np.array(self.memory).reshape((len(self.memory),)+self.state_size)
        #print(target_f)
        #print("OOOO",target_f.shape)
        self.model.fit(minibatch, minibatch, epochs=5, verbose=1)
        self.memory.clear()




def main():
    env = retro.make(game='BustAMove-Snes', state=random.choice(ESTADOS))
    state_size = env.reset()[20:212,72:184].shape
    #disparar - izquierda - derecha - esperar
    action_size = 40
    agent = Q_Agent(state_size, action_size)
    print(state_size)
    memoria = env.em.get_state()
    estados_extra= deque(maxlen=20)
    done = False
    batch_size = 50
    episodes = 1000000
    nivelar = lambda x: 1.0 if x>127 else 0.0
    func = lambda x: x/255.0
    print(nivelar(255),nivelar(122))
    num_accion = 0
    try:
        agent.load('pesosauto.h5')
        pass
    except:
        print("error")
    for e in range(episodes):
        tiempo_espera = 1
        state = env.reset()[20:212,72:184]
        #env.data.set_value('arrow2', 4)
        #state, reward, done, _a = env.step(agent.toBinary(3))
        #state, reward, done, _a = env.step(agent.toBinary(3))
        next_state, reward, done, _a = env.step(agent.toBinary(3))
        current_score = _a['bubbles']
        done=False
        state = func(state).reshape((1,state.shape[0],state.shape[1],state.shape[2]))
        suma = 0
        agent.model.predict([state])
        while not done:
            #env.render()
            #time.sleep(tiempo_espera)

            #angulo buscado
            action = agent.act(state)
            #no hay angulo 64 en el juego
            if random.random()>0.95:
                estados_extra.append(env.em.get_state())

            shoot_angle = 3*action+4
            if shoot_angle==64:
                shoot_angle=65
            next_state, reward, done, _a = env.step(agent.toBinary(3))
            #print("AAAA",_a['arrow2'])
            #print("BBBB", _a['arrow'])
            #env.render()
            #time.sleep(2.5)
            env.data.set_value('arrow2', shoot_angle)
            next_state, reward, done, _a = env.step(agent.toBinary(3))
            angle = _a['arrow']

            #print("VOY A DISPARAR A",shoot_angle)
            #me muevo al angulo deseado
            #env.data.set_value('arrow', shoot_angle)
            #next_state, reward, done, _a = env.step(agent.toBinary(3))

            #env.render()
            #print("AAAA", _a['arrow2'])
            #print("BBBB", _a['arrow'])

            #time.sleep(2)

            while angle!=shoot_angle:
                # time.sleep(0.008)
                #time.sleep(tiempo_espera)
                #env.render()
                #print("busco ",shoot_angle,"soy ",angle)
                if angle < shoot_angle:
                    next_state, reward, done, _a = env.step(agent.toBinary(2))
                elif angle > shoot_angle:
                    next_state, reward, done, _a = env.step(agent.toBinary(1))
                else:
                    break
                angle = _a['arrow']
            #disparo
            next_state, reward, done, _a = env.step(agent.toBinary(0))
            #time.sleep(tiempo_espera)
            #espero a poder disparar nuevamente
            #time.sleep(0.008)
            #time.sleep(tiempo_espera)
            next_state, reward, done, _a = env.step(agent.toBinary(3))
            frames = 0
            while _a['ready_to_fire']==60963:
                frames+=1
                #time.sleep(0.008)
                #time.sleep(tiempo_espera)
                #env.render()
                next_state, reward, done, _a = env.step(agent.toBinary(3))
                if done:
                    break

            #print(frames)

            #env.data.set_variable('arrow',10)

            #env.render()
            next_score = _a['bubbles']
            #env.data
            #if next_score-current_score==0:
            #    reward = -1.0
            #else:
            #    reward = math.log(next_score-current_score)
            recompensa = next_score-current_score-1
            suma += recompensa
            #print(recompensa)
            #print(recompensa)
            current_score=next_score
            #env.data.set_value('arrow2', 4)
            for i in range(0, 12+random.randint(0,2)):
                env.step(agent.toBinary(3))
            next_state, reward, done, _a = env.step(agent.toBinary(3))
            if recompensa!=-1:
                for i in range(0,35):
                    env.step(agent.toBinary(3))
                next_state, reward, done, _a = env.step(agent.toBinary(3))
            #env.render()

            if done:
                #print("SUMA:",suma)
                recompensa  =-10
                #print("Final score:",current_score)

            #print(current_score)
            #print(_a)
            #reward = _a['score_jyuu']
            next_state = next_state[20:212,72:184]

            next_state = func(next_state).reshape(state.shape)

            #print("mi reward es ",recompensa)
            agent.remember(state)
            if len(agent.memory)>1000:
                agent.replay(batch_size)
            state = next_state
        #print("replay")

        agent.update_target_model()
        #print("termino")
        print(e,":")
        print("epsilon: ",agent.epsilon)
        print("gamma: ", 1.0-agent.gammaComp)
        if True:
            if random.random()<float(len(estados_extra))/(len(estados_extra)+10*len(ESTADOS)):
                env.initial_state=random.choice(estados_extra)
            else:
                env.load_state(random.choice(ESTADOS))
        if e % 5 == 0:
            print("guarda3")
            agent.save("pesosauto.h5")



if __name__ == '__main__':
    main()
