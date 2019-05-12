from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D , LSTM, MaxPooling2D
from keras.optimizers import Adam
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

nivelar = lambda x: 1.0 if x > 127 else 0.0
func = np.vectorize(nivelar)
mutar = lambda x,y: x+(random.random()-0.5)
func_mut = np.vectorize(mutar)

class Agent:
    def __init__(self,state_size,action_size):
        self.state_size = state_size
        self.action_size = action_size

        #RAZON DE APRENDIZAJE
        self.learning_rate = 0.05
        #MODELO CREADO
        self.model =self._build_model()

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        print("guarda3")
        self.model.save_weights(name)

    def _build_model(self):
        model = Sequential()

        #model.add(Conv2D(15,input_shape=self.state_size,kernel_size=(5,5),strides=(2,2),activation='relu'))
        #model.add(Flatten())
        #model.add(Dense(200, activation='relu'))
        #model.add(Dense(30, activation='relu'))
        #model.add(Dense(self.action_size, activation='linear'))
        model.add(MaxPooling2D(pool_size=(3,3),input_shape=self.state_size))
        model.add(Conv2D(10,kernel_size=(3,3),strides=(2,2)))
        model.add(Flatten())
        model.add(Dense(200, activation='hard_sigmoid'))
        model.add(Dense(2000, activation='hard_sigmoid'))
        model.add(Dense(self.action_size, activation='hard_sigmoid'))
        model.summary()
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model


    def act(self, state):
        act_values = self.model.predict(state)
        accion = np.argmax(act_values[0])
        #print(accion,act_values[0][accion])
        return accion


    def toBinary(self,action):
        if action==3:
            return np.zeros(12)
        accion = [0,6,7][action]
        return np.concatenate((np.zeros(accion), np.array([1]), np.zeros(11 - accion)))

    def evaluate(self,env):
        state = env.reset()[23:207, 72:182]
        current_score = 0.0
        done = False

        state = func(state).reshape((1, state.shape[0], state.shape[1], state.shape[2]))
        frames = 0

        self.model.predict([state])
        while not done:

            action = self.act(state)
            # no hay angulo 64 en el juego

            shoot_angle = 4 * action + 4
            if shoot_angle == 64:
                shoot_angle = 65
            next_state, reward, done, _a = env.step(self.toBinary(3))

            env.data.set_value('arrow2', shoot_angle)
            next_state, reward, done, _a = env.step(self.toBinary(3))
            angle = _a['arrow']

            while angle != shoot_angle:

                if angle < shoot_angle:
                    next_state, reward, done, _a = env.step(self.toBinary(2))
                elif angle > shoot_angle:
                    next_state, reward, done, _a = env.step(self.toBinary(1))
                else:
                    break
                angle = _a['arrow']

            next_state, reward, done, _a = env.step(self.toBinary(0))

            next_state, reward, done, _a = env.step(self.toBinary(3))

            while _a['ready_to_fire'] == 60963:
                frames+=1
                next_state, reward, done, _a = env.step(self.toBinary(3))
                if done:
                    break

            next_score = _a['bubbles']

            current_score = next_score
            # env.render()
            if done:
                return current_score,current_score-frames*(0.001)
                #recompensa = -10
                #print("Final score:", current_score)
            # print(current_score)
            # print(_a)
            # print(reward)
            # reward = _a['score_jyuu']
            next_state = next_state[23:207, 72:182]
            next_state = func(next_state).reshape(state.shape)
            # print("mi reward es ",recompensa)
            state = next_state
        # print("replay")

        # print("termino")

    def copy_weights(self,target):
        self.model.set_weights(target.model.get_weights())

    def mutate(self,mutation_rate,mutation_value=1.0):

        pesos = self.model.get_weights()
        for i in range(len(pesos)):
            shape = pesos[i].shape
            new_pesos = pesos[i].flatten()
            mask = np.zeros(len(new_pesos),dtype=int)
            #print("mask",mask.shape)
            mask[:int(len(new_pesos)*mutation_rate)]=1
            np.random.shuffle(mask)
            randomized = np.random.rand(len(new_pesos))-np.ones(len(new_pesos))*0.5
            randomized = randomized*mutation_value
            #print("randomized",randomized.shape)
            mask = mask * randomized
            #print("random mask",mask.shape)
            #print(mask)
            new_pesos = new_pesos + mask
            #new_pesos = func_mut(new_pesos,mutation_rate)
            pesos[i] = new_pesos.reshape(shape)
        self.model.set_weights(pesos)

def main():
    env = retro.make(game='BustAMove-Snes', state=random.choice(ESTADOS))
    agents_num = 30
    very_little_random =3
    little_random = 6
    big_random = 20
    state_size = env.reset()[23:207,72:182].shape
    #disparar - izquierda - derecha - esperar
    action_size = 30
    agents = [Agent(state_size, action_size) for x in range(agents_num)]
    done = False
    episodes = 1000000
    print(nivelar(255),nivelar(122))
    try:
        agents[0].load('pesosconvo.h5')
        pass
    except:
        print("error")
    for e in range(episodes):
        for i in range(1,1+little_random):
            agents[i].copy_weights(agents[0])
            agents[i].mutate(0.01)
        for i in range(1+little_random,1+little_random+big_random):
            agents[i].copy_weights(agents[0])
            agents[i].mutate(0.1)
        for i in range(1 + little_random+big_random, 1 + little_random + big_random+very_little_random):
            agents[i].copy_weights(agents[0])
            agents[i].mutate(0.005,0.1)

        best_agent = 0
        _,best_score = agents[0].evaluate(env)
        print("puntaje",0,_, best_score)
        for i in range(1,agents_num):

            real_score,my_score = agents[i].evaluate(env)
            print("puntaje", i, real_score,my_score)
            if my_score > best_score:
                best_agent = i
                best_score = my_score
        print("gen %d:"%(e),best_score)
        agents[0].copy_weights(agents[best_agent])
        if e%50 == 0:
            env.load_state(random.choice(ESTADOS))
            agents[0].save("pesosconvo.h5")






if __name__ == '__main__':
    main()
