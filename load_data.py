import torch
import torch.utils.data as data
import numpy as np


class getReader():
    def __init__(self, path):
        self.path = path

    def readData(self):
        skill_list = []
        problem_list = []
        ans_list = []
        error_feedback_list = []
        partial_score_list = []
        split_char = ','

        read = open(self.path, 'r')
        for index, line in enumerate(read):
            if index % 7 == 0:
                pass
            elif index % 7 == 1:
                problems = line.strip().split(split_char)
                problems = list(map(int, problems))
                problem_list.append(problems)
            elif index % 7 == 2:
                skills = line.strip().split(split_char)
                skills = list(map(int, skills))
                skill_list.append(skills)
            elif index % 7 == 3:
                ans = line.strip().split(split_char)
                ans = list(map(float, ans))
                ans = [int(x) for x in ans]
                ans_list.append(ans)
            elif index % 7 == 4:
                error_feedback = line.strip().split(split_char)
                error_feedback = list(map(int, error_feedback))
                error_feedback_list.append(error_feedback)
            elif index % 7 == 5:
                partial_score = line.strip().split(split_char)
                partial_score = list(map(float, partial_score))
                partial_score_list.append(partial_score)

        read.close()
        return problem_list, skill_list, ans_list, error_feedback_list, partial_score_list


class KT_Dataset(data.Dataset):
    def __init__(self, problem_max, problem_list, skill_list, ans_list, error_feedback_list, partial_score_list, min_problem_num, max_problem_num):
        self.problem_max = problem_max
        self.min_problem_num = min_problem_num
        self.max_problem_num = max_problem_num
        self.problem_list, self.ans_list = [], []
        self.skill_list = []
        self.error_feedback_list = []
        self.partial_score_list = []

        for (problem, ans, skill, error_feedback, partial_score) in zip(problem_list, ans_list, skill_list, error_feedback_list, partial_score_list):
            num = len(problem)
            if num < min_problem_num:
                continue
            elif num > max_problem_num:
                segment = num // max_problem_num
                now_problem = problem[num - segment * max_problem_num:]
                now_ans = ans[num - segment * max_problem_num:]
                now_skill = skill[num - segment * max_problem_num:]
                now_error_feedback = error_feedback[num - segment * max_problem_num:]
                now_partial_score = partial_score[num - segment * max_problem_num:]

                if num > segment * max_problem_num:
                    self.problem_list.append(problem[:num - segment * max_problem_num])
                    self.ans_list.append(ans[:num - segment * max_problem_num])
                    self.skill_list.append(skill[:num - segment * max_problem_num])
                    self.error_feedback_list.append(error_feedback[:num - segment * max_problem_num])
                    self.partial_score_list.append(partial_score[:num - segment * max_problem_num])

                for i in range(segment):
                    item_problem = now_problem[i * max_problem_num:(i + 1) * max_problem_num]
                    item_ans = now_ans[i * max_problem_num:(i + 1) * max_problem_num]
                    self.problem_list.append(item_problem)
                    self.ans_list.append(item_ans)
                    self.skill_list.append(now_skill[i * max_problem_num:(i + 1) * max_problem_num])
                    self.error_feedback_list.append(now_error_feedback[i * max_problem_num:(i + 1) * max_problem_num])
                    self.partial_score_list.append(now_partial_score[i * max_problem_num:(i + 1) * max_problem_num])

            else:
                item_problem = problem
                item_ans = ans
                self.problem_list.append(item_problem)
                self.ans_list.append(item_ans)
                self.skill_list.append(skill)
                self.error_feedback_list.append(error_feedback)
                self.partial_score_list.append(partial_score)

    def __len__(self):
        return len(self.problem_list)

    def __getitem__(self, index):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        now_problem = self.problem_list[index]
        now_problem = np.array(now_problem)
        now_skill = self.skill_list[index]
        now_ans = self.ans_list[index]
        now_error_feedback = self.error_feedback_list[index]
        now_partial_score = self.partial_score_list[index]

        use_problem = np.zeros(self.max_problem_num, dtype=int)
        use_ans = np.zeros(self.max_problem_num, dtype=int)
        use_skill = np.zeros(self.max_problem_num, dtype=int)
        use_error_feedback = np.zeros(self.max_problem_num, dtype=int)
        use_partial_score = np.zeros(self.max_problem_num, dtype=float)
        use_mask = np.zeros(self.max_problem_num, dtype=int)

        num = len(now_problem)
        use_problem[-num:] = now_problem
        use_ans[-num:] = now_ans
        use_skill[-num:] = now_skill
        use_error_feedback[-num:] = now_error_feedback
        use_partial_score[-num:] = now_partial_score

        next_ans = use_ans[1:]
        next_problem = use_problem[1:]
        next_skill = use_skill[1:]
        next_error_feedback = use_error_feedback[1:]
        next_partial_score = use_partial_score[1:]

        last_ans = use_ans[:-1]
        last_problem = use_problem[:-1]
        last_skill = use_skill[:-1]
        last_error_feedback = use_error_feedback[:-1]
        last_partial_score = use_partial_score[:-1]

        mask = np.zeros(self.max_problem_num - 1, dtype=int)
        mask[-num + 1:] = 1

        use_mask[-num:] = 1
        last_mask = use_mask[:-1]
        next_mask = use_mask[1:]

        last_problem = torch.from_numpy(last_problem).to(device).long()
        next_problem = torch.from_numpy(next_problem).to(device).long()
        last_ans = torch.from_numpy(last_ans).to(device).long()
        next_ans = torch.from_numpy(next_ans).to(device).float()

        last_skill = torch.from_numpy(last_skill).to(device).long()
        next_skill = torch.from_numpy(next_skill).to(device).long()

        last_error_feedback = torch.from_numpy(last_error_feedback).to(device).long()
        next_error_feedback = torch.from_numpy(next_error_feedback).to(device).long()

        last_partial_score = torch.from_numpy(last_partial_score).to(device).float()
        next_partial_score = torch.from_numpy(next_partial_score).to(device).float()

        return last_problem, last_skill, last_ans, next_problem, next_skill, next_ans, last_error_feedback, next_error_feedback, last_partial_score, next_partial_score, torch.tensor(mask == 1).to(device)


def getLoader(problem_max, pro_path, skill_path, batch_size, is_train, min_problem_num, max_problem_num):
    problem_list, skill_list, ans_list, error_feedback_list, partial_score_list = getReader(pro_path).readData()
    dataset = KT_Dataset(problem_max, problem_list, skill_list, ans_list, error_feedback_list, partial_score_list, min_problem_num, max_problem_num)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)
    return loader
