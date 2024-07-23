from src.eval.utils.metrics import normalize_answer, f1_score
from typing import Tuple, Dict


class EvaluatePlanning:
    def reset(
            self,
            id: int,
            question: Dict[str, str],
            planning_name,
            pointer,
            planning_graph,
            task_name
    ) -> Tuple[bool, str, Dict]:
        ques, answer = question.get("question", None), question.get("answer", None)
        if ques is None or answer is None:
            return False, "Question not Valid", {}

        new_q = dict(
            id=id,
            step=0,
            task_name=task_name,
            planning_name=planning_name,
            human_counts=0,
            msg_status=0,  # 0: agent process, # 1: human to process
            status=0, # 0 initial, 1 in process, 2 finsh
            planning_status=planning_graph,
            pointer=pointer,
            task=(ques, answer),
            done=False,
            traj=[],
            messages=[],
            error=None,
            answer=None,
        )
        return True, "OK.", new_q

    def get_reward(self, info):
        if info['answer'] is not None and info['task'][1] is not None:
            pred = normalize_answer(info['answer'])
            gt = normalize_answer(info['task'][1])
            score = (pred == gt)
            return int(score)
        return 0

    def get_metrics(self, info):
        if info['answer'] is not None:
            pred = normalize_answer(info['answer'])
            gt = normalize_answer(info['task'][1])
            em = (pred == gt)
            f1 = f1_score(pred, gt)[0]
            return {'reward': em, 'em': em, 'f1': f1}
        return {'reward': 0, 'em': 0, 'f1': 0}
