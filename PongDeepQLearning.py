
import gym
import universe  # register the universe environments
import tensorflow as tf
import numpy as np
from collections import deque# Ordered collection with ends
import random
import sys
import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage

tf.enable_eager_execution()
warnings.filterwarnings('ignore')

#Setup hyperparameters

### MODEL HYPERPARAMETERS
state_size = [80, 80, 4]      # Our input is a stack of 4 frames hence 80x80x4 (Width, height, channels)
action_size = 3 # 3 possible actions
learning_rate =  0.00025      # Alpha (aka learning rate)

### TRAINING HYPERPARAMETERS
total_episodes = 5000           # Total episodes for training
max_steps = 500000              # Max possible steps in an episode
batch_size = 64                # Batch size

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability
decay_rate = 0.00001           # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.9                    # Discounting rate

### MEMORY HYPERPARAMETERS
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000          # Number of experiences the Memory can keep

### PREPROCESSING HYPERPARAMETERS
stack_size = 4                 # Number of frames stacked

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = True

# Define actions
up = [('KeyEvent', 'ArrowRight', True), ('KeyEvent', 'ArrowLeft', False)] # -> 0
down = [('KeyEvent', 'ArrowLeft', True), ('KeyEvent', 'ArrowRight', False)] # -> 1
still = [('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False)] # -> 2
default_action_index = 2
actions = [up, down, still]
encoded_actions = np.array(np.identity(len(actions),dtype=int).tolist())


# DQ Model
class DQLModel(tf.keras.Model):

    def __init__(self, name='DQLModel'):
        super(DQLModel, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation='relu',
                                            name="conv1")
        self.maxpool1 = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name="maxpool1")

        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation='relu',
                                            name="conv2")
        self.maxpool2 = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name="maxpool2")

        self.flatten1 = tf.keras.layers.Flatten(name="flatten1")
        self.fc1 = tf.keras.layers.Dense(128, activation='relu', name="fc1")
        self.out = tf.keras.layers.Dense(3, name="output")

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs, dtype=tf.float64)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten1(x)
        x = self.fc1(x)
        output = self.out(x)
        return output

# Tensorboard logging
global_step = tf.train.get_or_create_global_step()
log_dir = './train_log'
summary_writer = tf.contrib.summary.create_file_writer(
    log_dir, flush_millis=10000)

# Train Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

# Train loss logging
#train_loss = tf.keras.metrics.Mean(name='train_loss')

# DQL Model
dqlModel = DQLModel()

# Init Checkpoint
ckpt = tf.train.Checkpoint(step=tf.train.get_or_create_global_step(), optimizer=optimizer, model=dqlModel)
manager = tf.train.CheckpointManager(ckpt, './model_ckpts', max_to_keep=3)


# Preprocess state / observation image
def preprocess_frame(state):
    #import pdb
    #pdb.set_trace()
    state = state[35:195, 0:160, :]  # crop
    state = state[::2, ::2, 0]  # downsample by factor of 2 and get rid of G,B layer
    state[state == 144] = 0  # erase background (background type 1)
    state[state == 109] = 0  # erase background (background type 2)
    state[state != 0] = 1  # everything else (paddles, ball) just set to 1
    return state.astype('float64')

# Return actual action (to be pass in to env) and encoded action (use in our model)
def get_action(index):
    return actions[index], encoded_actions[index]


def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)

    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((80, 80), dtype=np.int) for i in range(stack_size)], maxlen=4)

        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames

def populate_memory(env, memory):
    for i in range(pretrain_length):
        # If it is first step
        if i == 0:
            state = env.reset()
            # There is a delay after we call reset, This is a workaround to wait until the game start and observation
            # is returned
            while state[0] is None:
                env_action, action = get_action(default_action_index)
                state, reward, done, info = env.step([env_action])
                reward, done = reward[0], done[0]
            assert len(state) == 1
            state, stacked_frames = stack_frames(None, state[0]['vision'], True)

        # env.render()
        # Get the next_state, the rewards, done by taking a random action
        choice = random.randint(1, len(actions)) - 1
        env_action, action = get_action(choice)
        next_state, reward, done, _ = env.step([env_action])
        reward, done = reward[0], done[0]
        assert len(next_state) == 1
        # Stack the frames
        next_state, stacked_frames = stack_frames(stacked_frames, next_state[0]['vision'], False)

        # If the episode is finished (opponent score 21)
        if done:
            # We finished the episode
            next_state = np.zeros(state.shape)

            # Add experience to memory
            memory.add((state, action, reward, next_state, done))


            # Start a new episode
            state = env.reset()
            while state[0] is None:
                env_action, action = get_action(default_action_index)
                state, reward, done, info = env.step([env_action])
                reward, done = reward[0], done[0]

            # Stack the frames
            assert len(state) == 1
            state, stacked_frames = stack_frames(stacked_frames, state[0]['vision'], True)

        else:
            # Add experience to memory
            memory.add((state, action, reward, next_state, done))

            # Our new state is now the next_state
            state = next_state


"""
This function will do the part
With ϵϵ select a random action atat, otherwise select at=argmaxaQ(st,a)
"""
def predict_action(explore_start, explore_stop, decay_rate, decay_step, state_to_predict, actions, model):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    exp_tradeoff = np.random.rand()

    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if (explore_probability > exp_tradeoff):
        # Make a random action (exploration)
        choice = random.randint(1, len(actions)) - 1
        env_action, action = get_action(choice)

    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        state_to_predict = state_to_predict.reshape((1, *(state_to_predict.shape)))
        Qs = model(state_to_predict)

        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        env_action, action = get_action(choice)

    return env_action, action, explore_probability

# Loss function
def loss_fn(actions_taken, model_outputs, target_Q):
    #Q = tf.reduce_sum(actions_taken * model_outputs)
    Q = tf.reduce_sum(tf.multiply(actions_taken, model_outputs), 1)
    loss = tf.reduce_mean(tf.square(target_Q - Q))
    return loss

# Learning Step
def learning_step(targets, states, actions_taken, model):
    with tf.GradientTape() as tape:
        model_outputs = model(states)
        loss = loss_fn(actions_taken, model_outputs, targets)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables), global_step=global_step)

    #train_loss(loss)

    with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10,
                                                                                               global_step=global_step):
        tf.contrib.summary.scalar('Loss', loss)

    return loss

class Memory():
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        indexs = np.random.choice(np.arange(buffer_size),
                                  size=batch_size,
                                  replace=False)
        return [self.buffer[i] for i in indexs]

def run_train(env, model):
    rewards_list = []
    # Initialize the decay rate (that will use to reduce epsilon)

    memory = Memory(memory_size)
    #Initialize memory
    populate_memory(env, memory)

    for episode in range(total_episodes):
        # Set step to 0
        step = 0
        # Initialize the rewards of the episode
        episode_rewards = []

        # Make a new episode and observe the first state
        state = env.reset()
        while state[0] is None:
            env_action, action = get_action(default_action_index)
            state, reward, done, info = env.step([env_action])
            reward, done = reward[0], done[0]

        # Remember that stack frame function also call our preprocess function.
        original_state_shape = state[0]['vision'].shape
        assert len(state) == 1
        state, stacked_frames = stack_frames(None, state[0]['vision'], True)

        while step < max_steps:
            step += 1

            # Get decay step from global step to support resume training
            decay_step = global_step.numpy()

            # Predict the action to take and take it
            action_env, action, explore_probability = predict_action(explore_start, explore_stop, decay_rate,
                                                                     decay_step, state, actions, model)

            # Perform the action and get the next_state, reward, and done information
            next_state, reward, done, _ = env.step([action_env])
            reward, done = reward[0], done[0]

            if episode_render:
                env.render()

            # Add the reward to total reward
            episode_rewards.append(reward)

            # If the game is finished
            if done:
                # The episode ends so no next state
                next_state = np.zeros(original_state_shape, dtype=np.int)

                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                # Set step = max_steps to end the episode
                step = max_steps

                # Get the total reward of the episode
                total_reward = np.sum(episode_rewards)

                #print('Episode: {}'.format(episode),
                # 'Total reward: {}'.format(total_reward),
                # 'Explore Probability: {:.4f}'.format(explore_probability))
                episod_tensor = tf.convert_to_tensor(episode, dtype=tf.int64)
                with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar('Explore Probability', tf.convert_to_tensor(explore_probability, dtype=tf.float32), step=episod_tensor)
                    tf.contrib.summary.scalar('Total Reward', tf.convert_to_tensor(total_reward, dtype=tf.float32), step=episod_tensor)

                rewards_list.append((episode, total_reward))

                # Store transition <st,at,rt+1,st+1> in memory D
                memory.add((state, action, reward, next_state, done))

            else:
                assert len(next_state) == 1
                # Stack the frame of the next_state
                next_state, stacked_frames = stack_frames(stacked_frames, next_state[0]['vision'], False)

                # Add experience to memory
                memory.add((state, action, reward, next_state, done))

                # st+1 is now our current state
                state = next_state

            ### LEARNING PART
            # Obtain random mini-batch from memory
            batch = memory.sample(batch_size)
            states_mb = np.array([each[0] for each in batch], ndmin=3)
            actions_mb = np.array([each[1] for each in batch])
            rewards_mb = np.array([each[2] for each in batch])
            next_states_mb = np.array([each[3] for each in batch], ndmin=3)
            dones_mb = np.array([each[4] for each in batch])

            target_Qs_batch = []

            # Get Q values for next_state
            Qs_next_state = model(next_states_mb)

            # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
            for i in range(0, len(batch)):
                terminal = dones_mb[i]

                # If we are in a terminal state, only equals reward
                if terminal:
                    target_Qs_batch.append(rewards_mb[i])

                else:
                    target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                    target_Qs_batch.append(target)

            targets_mb = np.array(target_Qs_batch)

            loss = learning_step(targets_mb, states_mb, actions_mb, model)

        # Save model every 5 episodes
        if episode % 5 == 0:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

def run_test(env, model, number_episod=10):
    for i in range(number_episod):
        state = env.reset()
        while state[0] is None:
            env_action, action = get_action(default_action_index)
            state, reward, done, info = env.step([env_action])
            reward, done = reward[0], done[0]

        assert len(state) == 1
        state, stacked_frames = stack_frames(None, state[0]['vision'], True)

        while done is False:
            state_to_predict = state.reshape((1, *(state.shape)))
            Qs = model(state_to_predict)

            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            env_action, action = get_action(choice)
            state, reward, done, info = env.step([env_action])

            reward, done = reward[0], done[0]
            assert len(state) == 1
            next_state, stacked_frames = stack_frames(stacked_frames, state[0]['vision'], False)
            state = next_state

    env.close()



if __name__ == '__main__':

    env = gym.make('gym-core.PongDeterministic-v3')
    #env = gym.make("Pong-v0")
    env.configure(remotes=1)

    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("No previous checkpoint saved. ")

    try:
        if len(sys.argv) >= 2 and sys.argv[1] == 'test':
            run_test(env, dqlModel, int(sys.argv[2]) if len(sys.argv) == 3 else 10)
        else:
            print('Run training...')
            run_train(env, dqlModel)
    except ValueError:
        print("Please pass in correct parameter.")



