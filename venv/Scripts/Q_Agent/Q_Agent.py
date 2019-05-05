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
        self.memory = deque(maxlen=300)
        #RAZON DE DESCUENTO
        self.gamma = 0.99

        #RAZON DE EXPLORACION
        self.epsilon = 1.0
        #EPSILON MINIMO
        self.min_epsilon = 0.05
        #DESCUENTO DEL EPSILON
        self.decay_epsilon = 0.9999
        #DESCUENTO DEL DESCUENTO
        #self.decay_decay = 0.99999
        #RAZON DE APRENDIZAJE
        self.learning_rate = 0.3
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
        model.add(Conv2D(10,kernel_size=(3,3)))
        model.add(Flatten())
        model.add(Dense(300,activation='relu'))
        model.add(Dense(800,activation='relu'))
        model.add(Dense(800, activation='relu'))
        model.add(Dense(800, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.summary()
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            accion =random.randrange(self.action_size)
        else:
            act_values = self.model.predict(state)
            accion = np.argmax(act_values[0])
        return accion


    def toBinary(self,action):
        if action==3:
            return np.zeros(12)
        accion = [0,6,7][action]
        return np.concatenate((np.zeros(accion), np.array([1]), np.zeros(11 - accion)))
    def replay(self, batch_size):
        if(len(self.memory)<batch_size):
            return
        minibatch = random.sample(self.memory,batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward +self.gamma * np.amax(self.model.predict(next_state))
            target_f = self.model.predict(state,steps=1)

            target_f[0][action] = target
            #print("OOOO",target_f.shape)
            #print("voy a hacer la wea")
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.decay_epsilon




def main():
    env = retro.make(game='BustAMove-Snes', state=random.choice(ESTADOS))

    state_size = env.reset()[23:207,72:182].shape
    #disparar - izquierda - derecha - esperar
    action_size = 60
    agent = Q_Agent(state_size, action_size)
    print(state_size)
    done = False
    batch_size = 50
    episodes = 1000000
    try:
        agent.load('pesos.h5')
        pass
    except:
        print("error")
    for e in range(episodes):
        tiempo_espera = 0.01
        state = env.reset()[23:207,72:182]
        current_score = 0.0
        done=False
        state = state.reshape((1,state.shape[0],state.shape[1],state.shape[2])).astype('float16')/256.0
        agent.model.predict([state])
        while not done:
            #env.render()
            #time.sleep(tiempo_espera)
            #angulo buscado
            action = agent.act(state)
            #no hay angulo 64 en el juego

            if action==30:
                action=31
            shoot_angle = 2*action+4

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

            #time.sleep(4)

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
            while _a['ready_to_fire']==60963:
                #time.sleep(0.008)
                #time.sleep(tiempo_espera)
                #env.render()
                next_state, reward, done, _a = env.step(agent.toBinary(3))
                if done:
                    break
            #weaaa = env.data.get_variable('arrow')
            #print(type(weaaa))

            #env.data.set_variable('arrow',10)
            #env.render()
            next_score = _a['bubbles']
            #env.data
            #print(weaaa)
            #tens = (_a['score_jyuu'] // 16) * 10
            #hundreds = (_a['score_hyaku'] % 16) * 100
            #thousands = ((_a['score_hyaku'] // 16)) * 1000
            #thousands = ((_a['score_hyaku'] // 16)) * 1000
            #ten_thousands = (_a['score_man'] % 16) * 10000
            #hundred_thousands = ((_a['score_man']// 16)) * 100000
            #next_score = tens + hundreds + thousands + ten_thousands + hundred_thousands
            #if next_score-current_score==0:
            #    reward = -1.0
            #else:
            #    reward = math.log(next_score-current_score)
            current_score=next_score
            env.render()
            if done:
                reward=-math.log(10000)
                print("Final score:",current_score)
            #print(current_score)
            #print(_a)
            #print(reward)
            #reward = _a['score_jyuu']
            next_state = next_state[23:207,72:182]
            next_state = next_state.reshape(state.shape).astype('float16')/256.0
            agent.remember(state, action, reward, next_state, done)
            state = next_state
        #print("replay")

        agent.replay(batch_size)
        #print("termino")
        print(e,":")
        print(agent.epsilon)
        if e % 10 == 0:
            env.load_state(random.choice(ESTADOS))
            print("guarda3")
            agent.save("pesos.h5")



if __name__ == '__main__':
    main()
