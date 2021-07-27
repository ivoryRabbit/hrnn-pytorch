import torch
from torch import nn
from torch.nn.functional import one_hot


class HGRU4REC(nn.Module):
    def __init__(
        self,
        device,
        input_size,
        output_size,
        hidden_dim,
        hidden_act="tanh",
        final_act="tanh",
        dropout_init=0.1,
        dropout_user=0.2,
        dropout_session=0.2,
        embedding_dim=-1,
        fft_all=False
    ):
        super(HGRU4REC, self).__init__()
        # arguments
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.hidden_act = hidden_act
        self.final_act = final_act
        self.dropout_init = dropout_init
        self.dropout_user = dropout_user
        self.dropout_session = dropout_session
        self.embedding_dim = embedding_dim
        self.fft_all = fft_all

        # layers
        self.u2s = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.get_activation(self.hidden_act),
            nn.Dropout(self.dropout_init)
        )
        self.s2o = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_size),
            self.get_activation(self.final_act),
        )
        self.dropout_user_layer = nn.Dropout(self.dropout_user)
        self.dropout_session_layer = nn.Dropout(self.dropout_session)

        if self.embedding_dim == -1:
            self.session_gru = nn.GRUCell(self.input_size, self.hidden_dim)
        else:
            self.look_up = nn.Embedding(self.input_size, self.embedding_dim)
            self.session_gru = nn.GRUCell(self.embedding_dim, self.hidden_dim)
        self.user_gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)

        if self.fft_all:
            self.fft = nn.Linear(self.hidden_dim, self.input_size, bias=False)
        self = self.to(self.device)

    def forward(self, input, session_repr, session_mask, user_repr, user_mask):
        if self.embedding_dim == -1:
            embedded = self.onehot_encode(input)
        else:
            embedded = input.unsqueeze(0)
            embedded = self.look_up(embedded)

        # update user representative only when a new session updates
        user_repr_updt = self.user_gru(session_repr, user_repr)
        user_repr_updt = self.dropout_user_layer(user_repr_updt)
        user_repr = session_mask * user_repr_updt + (
                    1 - session_mask) * user_repr  # (batch_size, hidden_dim)

        # reset user representative for new user
        user_repr = user_mask * self.mask_zeros(user_repr) + (1 - user_mask) * user_repr

        # initialize session representative when a new session starts
        session_repr_init = self.u2s(user_repr)
        session_repr = session_mask * session_repr_init + (
                    1 - session_mask) * session_repr  # (batch_size, hidden_dim)

        # reset session representative for new user
        session_repr = user_mask * self.mask_zeros(session_repr) + (1 - user_mask) * session_repr
        session_repr = self.session_gru(embedded, session_repr)
        session_repr = self.dropout_session_layer(session_repr)

        score = self.s2o(session_repr)
        return score, session_repr, user_repr

    def onehot_encode(self, input):
        encoded = one_hot(input, num_classes=self.input_size).float()
        return encoded.to(self.device)

    def get_mask(self, mask_idx, batch_size):
        mask = torch.zeros(batch_size, self.hidden_dim)
        mask[mask_idx, :] = 1.0
        return mask.to(self.device)

    def mask_zeros(self, repr):
        return torch.zeros_like(repr).to(self.device)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(batch_size, self.hidden_dim)
        return hidden.to(self.device)

    def init_model(self, sigma):
        for p in self.parameters():
            if sigma > 0:
                p.data.uniform_(-sigma, sigma)
            elif len(p.size()) > 1:
                sigma_ = (6.0 / (p.size(0) + p.size(1))) ** 0.5
                if sigma == -1:
                    p.data.uniform_(-sigma_, sigma_)
                else:
                    p.data.uniform_(0, sigma_)

    def save(self, save_dir):
        model_state = dict(
            model=self.state_dict(),
            args=dict(
                input_size=self.input_size,
                output_size=self.output_size,
                hidden_dim=self.hidden_dim,
                hidden_act=self.hidden_act,
                final_act=self.final_act,
                dropout_init=self.dropout_init,
                dropout_user=self.dropout_user,
                dropout_session=self.dropout_session,
                embedding_dim=self.embedding_dim,
                fft_all=self.fft_all,
            )
        )
        torch.save(model_state, save_dir)

    def get_activation(self, act_name):
        if act_name == "tanh":
            return nn.Tanh()
        elif act_name == "relu":
            return nn.ReLU()
        elif act_name == "softmax":
            return nn.Softmax()
        elif act_name == "softmax_logit":
            return nn.LogSoftmax()
