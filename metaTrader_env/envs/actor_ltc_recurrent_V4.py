import torch.nn as nn
from matplotlib import pyplot as plt
from ncps.torch import CfC
from ncps.wirings import AutoNCP

import os
import numpy as np
import torch as T
import torch.optim as optim
from torch.distributions.categorical import Categorical


class SequencePPOMemory:
    """Memory buffer that stores episodes chronologically and returns 3D chunked sequences (batch_size, seq_len, features) for proper Recurrent PPO."""

    def __init__(self, batch_size, seq_len):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
        self.seq_len = seq_len

    def store_transition(self, state, action, prob, val, reward, done):
        self.states.append(state)
        self.probs.append(prob)
        self.actions.append(action)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

    def extract_advantages_and_chunks(self, gamma, gae_lambda):
        n_states = len(self.rewards)
        vals_arr = np.array(self.vals)
        rewards_arr = np.array(self.rewards)
        dones_arr = np.array(self.dones)

        # 1. Compute GAE Flat across the chronological sequences
        advantages = np.zeros(n_states, dtype=np.float32)
        for t in range(n_states):
            discount = 1
            a_t = 0
            for k in range(t, n_states):
                next_val = vals_arr[k + 1] if k < n_states - 1 else 0.0
                a_t += discount * (rewards_arr[k] + gamma * next_val * (1.0 - dones_arr[k]) - vals_arr[k])
                discount *= gamma * gae_lambda
            advantages[t] = a_t

        # 2. Chop memory cleanly into 3D Chunks: (N_Chunks, Seq_Len)
        n_chunks = n_states // self.seq_len
        valid_len = n_chunks * self.seq_len

        if n_chunks == 0:
            raise ValueError(
                f"Not enough experiences to form a single sequence chunk. Buffer size {n_states} < seq_len {self.seq_len}")

        st_chunks = np.array(self.states[:valid_len]).reshape(n_chunks, self.seq_len, -1)
        ac_chunks = np.array(self.actions[:valid_len]).reshape(n_chunks, self.seq_len)
        pr_chunks = np.array(self.probs[:valid_len]).reshape(n_chunks, self.seq_len)
        ad_chunks = advantages[:valid_len].reshape(n_chunks, self.seq_len)
        va_chunks = vals_arr[:valid_len].reshape(n_chunks, self.seq_len)

        # Normalize advantages safely
        adv_flat = ad_chunks.flatten()
        ad_chunks = (ad_chunks - adv_flat.mean()) / (adv_flat.std() + 1e-7)

        # 3. Batch the Chunks
        indices = np.arange(n_chunks, dtype=np.int32)
        # Todo 22/03/26:
        #               no need to shuffle
        np.random.shuffle(indices)  # We shuffle chunks, but NOT internal time steps!
        batches = [indices[i:i + self.batch_size] for i in range(0, n_chunks, self.batch_size)]

        return st_chunks, ac_chunks, pr_chunks, va_chunks, ad_chunks, batches

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.vals = []
        self.rewards = []
        self.dones = []


class ActorLTCNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, lr, ckpt_dir="../ppo", load_back_f=False):
        super(ActorLTCNetwork, self).__init__()

        self.checkpoint_file = os.path.join(ckpt_dir, "actor_torch_ppo_seq.pth")
        self.checkpoint_file_load = os.path.join(ckpt_dir + "/old",
                                                 "actor_torch_ppo_seq.pth") if load_back_f else self.checkpoint_file

        self.net1 = nn.Linear(input_dims, 64)
        self.dropout = nn.Dropout(p=0.35)

        self.ltc1 = CfC(64, AutoNCP(64, 32), return_sequences=True, batch_first=True)
        self.ltc2 = CfC(32, AutoNCP(32, 16), return_sequences=True, batch_first=True)

        self.final = nn.Linear(16, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, hx=None):
        # We expect natively 3D state: (batch_size, seq_len, features)
        if state.dim() == 1:
            state = state.unsqueeze(0).unsqueeze(1)  # Used for inference rollouts (batch=1, seq=1)
        elif state.dim() == 2:
            state = state.unsqueeze(1)

        out = T.relu(self.net1(state))
        out = self.dropout(out)

        if hx is None:
            hx = (None, None)


        out, h1 = self.ltc1(out, hx[0])
        # Chang here - added dropout here
        out = self.dropout(out)
        out, h2 = self.ltc2(out, hx[1])

        out = self.final(out)

        probs = nn.Softmax(dim=-1)(out)
        dist = Categorical(probs)

        hx_new = (h1, h2)
        return dist, hx_new

    def save_checkpoint(self, name=None):
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        if name is None:
            T.save(checkpoint, self.checkpoint_file)
        else:
            file = r"../ppo/actor_torch_ppo_seq_{}.pth".format(name)
            T.save(checkpoint, file)

    def load_checkpoint(self, name=None):
        if name is None:
            path = self.checkpoint_file_load
        else:
            path = os.path.normpath(str("../" + self.checkpoint_file_load.split(".")[-2]) + f"_{name}.pth")

        if os.path.exists(path):
            checkpoint = T.load(path, map_location=self.device)
            # Backwards compatibility check
            if 'model_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                self.load_state_dict(checkpoint)


class CriticLTCNetwork(nn.Module):
    def __init__(self, input_dims, lr, ckpt_dir="../ppo", load_back_f=False):
        super(CriticLTCNetwork, self).__init__()

        self.checkpoint_file = os.path.join(ckpt_dir, "critic_torch_ppo_seq.pth")
        self.checkpoint_file_load = os.path.join(ckpt_dir + "/old",
                                                 "critic_torch_ppo_seq.pth") if load_back_f else self.checkpoint_file

        self.net1 = nn.Linear(input_dims, 64)
        self.dropout = nn.Dropout(p=0.35)

        self.ltc1 = CfC(64, AutoNCP(64, 32), return_sequences=True, batch_first=True)
        self.ltc2 = CfC(32, AutoNCP(32, 16), return_sequences=True, batch_first=True)

        self.final = nn.Linear(16, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, hx=None):
        if state.dim() == 1:
            state = state.unsqueeze(0).unsqueeze(1)
        elif state.dim() == 2:
            state = state.unsqueeze(1)

        out = T.relu(self.net1(state))
        out = self.dropout(out)

        if hx is None:
            hx = (None, None)

        out, h1 = self.ltc1(out, hx[0])
        # Chang here - added dropout here
        out = self.dropout(out)
        out, h2 = self.ltc2(out, hx[1])

        out = self.final(out)
        value = out.squeeze(-1)  # (batch_size, seq_len)

        hx_new = (h1, h2)
        return value, hx_new

    def save_checkpoint(self, name=None):
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        if name is None:
            T.save(checkpoint, self.checkpoint_file)
        else:
            file = r"../ppo/actor_torch_ppo_seq_{}.pth".format(
                name)
            T.save(checkpoint, file)

    def load_checkpoint(self, name=None):
        if name is None:
            path = self.checkpoint_file_load
        else:
            path = os.path.normpath(str("../" + self.checkpoint_file_load.split(".")[-2]) + f"_{name}.pth")

        if os.path.exists(path):
            checkpoint = T.load(path, map_location=self.device)
            # Backwards compatibility check
            if 'model_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                self.load_state_dict(checkpoint)


class AgentRLTC_Seq:
    """True Recurrent PPO architecture solving the Batch-Shuffle conflict (BPTT Chunking enabled)."""

    def __init__(self, n_actions, input_dims, gamma=0.99, lr=0.0003,
                 gae_lambda=0.95, policy_clip=0.2, epochs=10, entropy_coef=0.01,
                 batch_size=32, seq_len=16, ckpt_dir="../ppo", load_back_file=False):
        self.gamma = gamma
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.epochs = epochs
        self.entropy_coef = entropy_coef

        self.actor = ActorLTCNetwork(n_actions, input_dims, lr, ckpt_dir=ckpt_dir, load_back_f=load_back_file)
        self.critic = CriticLTCNetwork(input_dims, lr, ckpt_dir=ckpt_dir, load_back_f=load_back_file)

        self.actor_scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(self.actor.optimizer, mode='min', factor=0.5,
                                                                      patience=5)
        self.critic_scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(self.critic.optimizer, mode='min', factor=0.5,
                                                                       patience=5)

        self.current_epoch_losses = []

        # Inference memory (1 step width rollouts)
        self.actor_hx = None
        self.critic_hx = None

        # Custom memory buffer that explicitly parses sequence chunks
        self.memory = SequencePPOMemory(batch_size=batch_size, seq_len=seq_len)
        self.actor_error = []
        self.critic_error = []
        self.actor_tot_error = []

    def remember(self, state, action, probs, vals, rewards, done):
        self.memory.store_transition(state, action, probs, vals, rewards, done)

    def save_model(self, name=None):
        print(">: ... saving sequence-based model ...")
        if name is None:
            self.actor.save_checkpoint()
            self.critic.save_checkpoint()
        else:
            self.actor.save_checkpoint(name)
            self.critic.save_checkpoint(name)

    def load_model(self, name=None):
        print(">: ... loading sequence-based model ...")
        self.actor.load_checkpoint(name)
        self.critic.load_checkpoint(name)

    def choose_action(self, observation):
        if not isinstance(observation, T.Tensor):
            state = T.tensor(observation, dtype=T.float32).to(self.actor.device)
        else:
            state = observation.type(T.float32).to(self.actor.device)

        dist, self.actor_hx = self.actor(state, self.actor_hx)
        #print(f"Shape of hx is {self.actor_hx}")
        value, self.critic_hx = self.critic(state, self.critic_hx)

        action = dist.sample()
        # Output exactly the flat items for memory storage
        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for i in range(self.epochs):
            self.current_epoch_losses = []

            # Generate perfectly chronological sequence chunks
            st_chunks, ac_chunks, pr_chunks, va_chunks, ad_chunks, batches = \
                self.memory.extract_advantages_and_chunks(self.gamma, self.gae_lambda)
            # changes here - we reset hidden states every episode but we keep it in every batch
            # since that's a continuation
            batch_actor_hx = None
            batch_critic_hx = None
            for batch in batches:
                # Shape is strictly (Batch_size, Seq_Len, Features)
                state = T.tensor(st_chunks[batch], dtype=T.float32).to(self.actor.device)
                action = T.tensor(ac_chunks[batch], dtype=T.float32).to(self.actor.device)
                old_prob = T.tensor(pr_chunks[batch], dtype=T.float32).to(self.actor.device)
                adv = T.tensor(ad_chunks[batch], dtype=T.float32).to(self.actor.device)
                values = T.tensor(va_chunks[batch], dtype=T.float32).to(self.actor.device)

                # Pushes entire chronological sequence mathematically in ONE swift operation
                # (Batch, Seq_Len, Out)
                # hidden state was commented
                dist, batch_actor_hx = self.actor(state, batch_actor_hx)
                critic_value, batch_critic_hx = self.critic(state, batch_critic_hx)

                # Calculate probabilities over the entire 3D object block
                new_prob = dist.log_prob(action)
                dist_entropy = dist.entropy()

                prob_ratio = T.exp(new_prob - old_prob)
                weighted_prob = adv * prob_ratio
                weighted_clipped = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * adv

                # Flatten the internal (Batch x Seq) losses
                actor_loss = -T.min(weighted_prob, weighted_clipped).mean()
                returns = adv + values
                critic_loss = (returns - critic_value).pow(2).mean()

                total_loss = actor_loss + (0.5 * critic_loss) - (self.entropy_coef * dist_entropy.mean())

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()

                total_loss.backward()

                # We've commented since it should be studied to now if we should adjust max_norm
                #nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                #nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)

                self.actor.optimizer.step()
                self.critic.optimizer.step()

                self.actor_error.append(actor_loss.item())
                self.critic_error.append(critic_loss.item())
                self.actor_tot_error.append(total_loss.item())
                self.current_epoch_losses.append(total_loss.item())

                # Initialize hidden state as NONE so CfC network inherently allocates
                # proper tensor memory (batch=B) automatically across the full Seq_Len window!
                # Shoudln't keep memory from batches
                batch_actor_hx = None
                batch_critic_hx = None
                #batch_actor_hx = [h.detach() for h in self.actor_hx]  # None
                #batch_critic_hx = [h.detach() for h in self.critic_hx]  # None

            #if self.current_epoch_losses:
            #    avg_loss = np.mean(self.current_epoch_losses)
            #    self.actor_scheduler.step(avg_loss)
            #    self.critic_scheduler.step(avg_loss)

        # Drop inference interactive states explicitly since epoch iteration trained
        # completely disconnected temporal states. Clean slate for next step iteration.
        self.actor_hx = [None, None]
        self.critic_hx = [None, None]

        self.memory.clear_memory()

    def plot_error(self, epoch):
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(self.actor_error)
        axs[0, 0].set_title('actor loss')
        axs[0, 1].plot(self.critic_error)
        axs[0, 1].set_title('critic loss')
        axs[1, 0].plot(self.actor_tot_error)
        axs[1, 0].set_title('Total loss')
        fig.tight_layout()
        fig.savefig(f'../images/ltc_agent_loss_seq_{epoch}.png')
        plt.close()
