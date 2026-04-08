import torch
import torch.nn as nn
import torch.nn.functional as F


class FDPKT(nn.Module):
    def __init__(self, pro_max, algo_ability_max, err_feedback_max, d, p,
                 use_response_enhancement=True,
                 use_response_change=True,
                 use_diagnosis_router=True):
        super(FDPKT, self).__init__()

        self.use_response_enhancement = use_response_enhancement
        self.use_response_change = use_response_change
        self.use_diagnosis_router = use_diagnosis_router

        self.pro_max = pro_max
        self.algo_ability_max = algo_ability_max

        self.pro_embed = nn.Parameter(torch.rand(pro_max, d))
        self.algo_ability_embed = nn.Parameter(torch.rand(algo_ability_max, d))
        self.ans_embed = nn.Parameter(torch.rand(2, d))
        self.err_feedback_embed = nn.Parameter(torch.rand(err_feedback_max, d))

        self.out = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(d, 1)
        )

        self.dropout = nn.Dropout(p=p)
        self.time_embed = nn.Parameter(torch.rand(200, d))
        self.ls_state = nn.Parameter(torch.rand(1, d))
        self.c_state = nn.Parameter(torch.rand(1, d))
        self.pro_state = nn.Parameter(torch.rand(199, d))
        self.algo_ability_state = nn.Parameter(torch.rand(199, d))

        self.obtain_pro_forget = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.Sigmoid()
        )
        self.obtain_pro_state = nn.Sequential(
            nn.Linear(3 * d, d)
        )

        if self.use_response_change:
            self.change_encoder = nn.Sequential(
                nn.Linear(3 * d, d)
            )

        if self.use_diagnosis_router:
            self.router = nn.Sequential(
                nn.Linear(2 * d, d),
                nn.Sigmoid()
            )

        self.coding_literacy_down = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.Sigmoid()
        )

        self.algo_ability_down = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.Sigmoid()
        )

        self.obtain_coding_literacy_forget = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.Sigmoid()
        )

        self.obtain_algo_ability_forget = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.Sigmoid()
        )
        self.obtain_algo_ability_state = nn.Sequential(
            nn.Linear(3 * d, d)
        )
        self.obtain_coding_literacy_state = nn.Sequential(
            nn.Linear(3 * d, d)
        )

        self.obtain_algo_ability_state_baseline = nn.Sequential(
            nn.Linear(2 * d, d)
        )
        self.obtain_coding_literacy_state_baseline = nn.Sequential(
            nn.Linear(2 * d, d)
        )

        self.akt_pro_diff = nn.Parameter(torch.rand(pro_max, 1))
        self.akt_pro_change = nn.Parameter(torch.rand(algo_ability_max, d))

    def forward(self, last_problem, last_algo_ability, last_ans, next_problem, next_algo_ability, next_ans,
                last_error_feedback, next_error_feedback, last_partial_score, next_partial_score):
        device = last_problem.device
        batch = last_problem.shape[0]
        seq = last_problem.shape[-1]

        next_pro_embed = F.embedding(next_problem, self.pro_embed) + F.embedding(next_algo_ability,
                                                                                self.algo_ability_embed) + F.embedding(
            next_problem, self.akt_pro_diff) * F.embedding(next_algo_ability, self.akt_pro_change)

        if self.use_response_enhancement:
            next_error_feedback_embed = F.embedding(next_error_feedback.long(), self.err_feedback_embed)
            weight = torch.where(next_ans >= 0.5, next_partial_score, 1 - next_partial_score)
            weight = weight.unsqueeze(-1)
            next_R = weight * next_error_feedback_embed
            next_X = next_pro_embed + next_R
        else:
            next_X = next_pro_embed + F.embedding(next_ans.long(), self.ans_embed)
            next_R = F.embedding(next_ans.long(), self.ans_embed)

        last_pro_time = torch.full((batch, self.pro_max), -1, dtype=torch.long).to(device)
        last_algo_ability_time = torch.zeros((batch, self.algo_ability_max)).to(device)
        algo_ability_state = self.algo_ability_state.unsqueeze(0).repeat(batch, 1, 1)
        coding_literacy_state = self.ls_state.repeat(batch, 1)
        last_algo_ability_state = self.algo_ability_state.unsqueeze(0).repeat(batch, 1, 1)

        batch_index = torch.arange(batch).to(device)
        coding_literacy_time_gap = torch.ones((batch, seq)).to(device)
        coding_literacy_time_gap_embed = F.embedding(coding_literacy_time_gap.long(), self.time_embed)

        res_p = []

        for now_step in range(seq):
            now_pro_embed = next_pro_embed[:, now_step]
            now_item_pro = next_problem[:, now_step]
            now_item_algo_ability = next_algo_ability[:, now_step]

            last_batch_algo_ability_time = last_algo_ability_time[batch_index, now_item_algo_ability]
            last_batch_algo_ability_state = algo_ability_state[batch_index, last_batch_algo_ability_time.long()]

            algo_ability_time_gap = now_step - last_batch_algo_ability_time
            algo_ability_time_gap_embed = F.embedding(algo_ability_time_gap.long(), self.time_embed)

            item_algo_ability_state_forget = self.obtain_algo_ability_forget(
                self.dropout(torch.cat([last_batch_algo_ability_state, algo_ability_time_gap_embed], dim=-1)))
            last_batch_algo_ability_state = last_batch_algo_ability_state * item_algo_ability_state_forget

            item_coding_literacy_state_forget = self.obtain_coding_literacy_forget(
                self.dropout(torch.cat([coding_literacy_state, coding_literacy_time_gap_embed[:, now_step]], dim=-1)))
            last_batch_coding_literacy_state = coding_literacy_state * item_coding_literacy_state_forget

            last_algo_ability_state[:, now_step] = last_batch_algo_ability_state

            final_state = torch.cat(
                [last_batch_coding_literacy_state, last_batch_algo_ability_state, now_pro_embed], dim=-1)

            P = torch.sigmoid(self.out(self.dropout(final_state))).squeeze(-1)
            res_p.append(P)
            now_item_R = next_R[:, now_step]

            gate_score = torch.ones(1, device=device)
            change_vector = torch.zeros_like(next_X[:, now_step])

            if self.use_diagnosis_router:
                gate_score = self.router(torch.cat([now_pro_embed, now_item_R], dim=-1))

            if self.use_response_change:
                last_batch_pro_time = last_pro_time[batch_index, now_item_pro]
                has_history = (last_batch_pro_time >= 0) & (last_batch_pro_time < now_step)
                last_pro_R = torch.where(has_history.unsqueeze(-1),
                                         next_R[batch_index, last_batch_pro_time.long()],
                                         torch.zeros_like(now_item_R))

                pro_diff = now_item_R - last_pro_R
                pro_prod = now_item_R * last_pro_R

                raw_change_input = torch.cat([now_item_R, pro_diff, pro_prod], dim=-1)
                change_vector = self.change_encoder(self.dropout(raw_change_input))

            if self.use_response_change and self.use_diagnosis_router:
                item_coding_literacy_obtain = self.obtain_coding_literacy_state(
                    self.dropout(torch.cat([last_batch_coding_literacy_state, next_X[:, now_step], change_vector], dim=-1)))
                item_coding_literacy_state = last_batch_coding_literacy_state + gate_score * torch.tanh(item_coding_literacy_obtain)
            elif self.use_response_change:
                item_coding_literacy_obtain = self.obtain_coding_literacy_state(
                    self.dropout(torch.cat([last_batch_coding_literacy_state, next_X[:, now_step], change_vector], dim=-1)))
                item_coding_literacy_state = last_batch_coding_literacy_state + torch.tanh(item_coding_literacy_obtain)
            elif self.use_diagnosis_router:
                item_coding_literacy_obtain = self.obtain_coding_literacy_state_baseline(
                    self.dropout(torch.cat([last_batch_coding_literacy_state, next_X[:, now_step]], dim=-1)))
                item_coding_literacy_state = last_batch_coding_literacy_state + gate_score * torch.tanh(item_coding_literacy_obtain)
            else:
                item_coding_literacy_obtain = self.obtain_coding_literacy_state_baseline(
                    self.dropout(torch.cat([last_batch_coding_literacy_state, next_X[:, now_step]], dim=-1)))
                item_coding_literacy_state = last_batch_coding_literacy_state + torch.tanh(item_coding_literacy_obtain)

            coding_literacy_state = item_coding_literacy_state
            algo_ability_get = next_X[:, now_step]

            if self.use_response_change and self.use_diagnosis_router:
                item_algo_ability_obtain = self.obtain_algo_ability_state(
                    self.dropout(torch.cat([last_batch_algo_ability_state, algo_ability_get, change_vector], dim=-1)))
                item_algo_ability_state = last_batch_algo_ability_state + (1 - gate_score) * torch.tanh(item_algo_ability_obtain)
            elif self.use_response_change:
                item_algo_ability_obtain = self.obtain_algo_ability_state(
                    self.dropout(torch.cat([last_batch_algo_ability_state, algo_ability_get, change_vector], dim=-1)))
                item_algo_ability_state = last_batch_algo_ability_state + torch.tanh(item_algo_ability_obtain)
            elif self.use_diagnosis_router:
                item_algo_ability_obtain = self.obtain_algo_ability_state_baseline(
                    self.dropout(torch.cat([last_batch_algo_ability_state, algo_ability_get], dim=-1)))
                item_algo_ability_state = last_batch_algo_ability_state + (1 - gate_score) * torch.tanh(item_algo_ability_obtain)
            else:
                item_algo_ability_obtain = self.obtain_algo_ability_state_baseline(
                    self.dropout(torch.cat([last_batch_algo_ability_state, algo_ability_get], dim=-1)))
                item_algo_ability_state = last_batch_algo_ability_state + torch.tanh(item_algo_ability_obtain)

            last_pro_time[batch_index, now_item_pro] = now_step
            last_algo_ability_time[batch_index, now_item_algo_ability] = now_step
            algo_ability_state[:, now_step] = item_algo_ability_state

        res_p = torch.vstack(res_p).T
        return res_p
