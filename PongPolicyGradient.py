import gym
import universe  # register the universe environments
import tensorflow as tf
import numpy as np

def prepro(I):
    """ prepro 210x160x3 (768, 1024, 3) uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195, 0:160, :]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel() #1 d array with 0 1


def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, len(r))):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r          


env = gym.make('gym-core.PongDeterministic-v3')

gamma = 0.99

up = [('KeyEvent', 'ArrowRight', True), ('KeyEvent', 'ArrowLeft', False)] # -> 0
down = [('KeyEvent', 'ArrowLeft', True), ('KeyEvent', 'ArrowRight', False)] # -> 1
still = [('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False)] # -> 2

actionArr = [down, up, still]

env.configure(remotes=1)  # automatically creates a local docker container
observation_n = env.reset()

#Construct TF Graph
imageInput = tf.placeholder(tf.float32, [None, 80*80])
rewardIput = tf.placeholder(tf.float32, [None, 1]) # N X 1 
actionInput = tf.placeholder(tf.int32, [None, 1]) # N X 1


hiddenLayerOutput = tf.layers.dense(inputs=imageInput, units=200, activation=tf.nn.relu)
outputLayerOutput = tf.layers.dense(inputs=hiddenLayerOutput, units=3) #keep linear activation?

#draw a sample action
sampleAction = tf.multinomial(logits=outputLayerOutput, num_samples=1) 
# this is actual action to be feeded
#return N X 1 X 3 array
onthotAction = tf.one_hot(actionInput, 3)
#reshape N X 3 array
finalAction = tf.reshape(onthotAction, [-1, 3])
#loss function N x 1 array
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=finalAction, logits=outputLayerOutput)
final_cross_entropy = tf.reshape(cross_entropy, [-1, 1])
#time reward and sum all loss across observation, both is N x 1 array
probs = tf.nn.softmax(logits=outputLayerOutput)
loss = tf.reduce_sum(rewardIput * final_cross_entropy)
tf.summary.scalar('Loss', loss)
#Gradient decent optimizing, there a number of optimizer to choose from
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.99)
#training op
optimize = optimizer.minimize(loss)

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver(max_to_keep=1)
model_path = "./pongCheckPointMoveCost/model.ckpt"

episodIndex = 0
noOfEpisod = 2000
# Using the context manager.
with tf.Session() as sess:
    #Initialize variables
    sess.run(tf.global_variables_initializer())
    # Restore model weights from previously saved model
    # saver.restore(sess, model_path)
    train_writer = tf.summary.FileWriter( './ponglogs', sess.graph)
    while (episodIndex < noOfEpisod): # no of training        
        episodIndex += 1        
        actionIndexArr = []
        rewardArr = []  
        done_ng = False 
        observationArr = None
        prevObservation = np.zeros((80*80))
        #imageNo = 0
        #imageArr = []
        while not done_ng:  # one round 21 points, one episod          
          #1. feed forward to get action
          #2. step with the action
          #3. get observation, proprocess and deduct previous observation
          #4. append/vstack the result in 3
          #5. append/vstack the reward next step for diff of observation in 4 - have to be align (current reward is for previous           
          #forward prog to get action  
          #observationCounter = observationCounter + 1 if prevObservation != None else 0
          #print("prevObservation ---->", prevObservation)
          if observation_n[0] != None: #game is running

            #get observation
            observation = prepro(observation_n[0]['vision']) 
            diffObservation = observation - prevObservation
            observationArr = diffObservation if observationArr is None else np.vstack((observationArr, diffObservation))

            actionIndexResult, outputLayerResult = sess.run([sampleAction, outputLayerOutput], 
                                 feed_dict={imageInput: diffObservation.reshape(1,-1)})
            actionIndex = actionIndexResult[0,0]

            actionIndexArr.append(actionIndex)
            action_n = [actionArr[actionIndex] for ob in observation_n]

            prevObservation = observation
            observation_n, reward_n, done_n, info = env.step(action_n)
            rewardArr.append(reward_n[0])
          else:
            action_n = [still] #dont do anything by default
            observation_n, reward_n, done_n, info = env.step(action_n)
          done_ng = done_n[0]
             
          env.render()
          #Done one episod

        rewardArr = discount_rewards(rewardArr)
        rewardMean = np.mean(rewardArr)
        normalisedReward = (rewardArr - rewardMean)/np.std(rewardArr)
        #reshape to feed in to tensorflow placeholder
        normalisedReward = np.vstack(normalisedReward)
        actionIndexArr = np.vstack(actionIndexArr)
        merge = tf.summary.merge_all()
        summary, _ = sess.run([merge, optimize], feed_dict={imageInput: observationArr,
                                     rewardIput: normalisedReward,
                                     actionInput: actionIndexArr})        

        train_writer.add_summary(summary, episodIndex)
        if noOfEpisod % 10 == 0 :
            # Save model weights to disk after N episod
            save_path = saver.save(sess, model_path)
            print("Model saved in file: ", save_path)
