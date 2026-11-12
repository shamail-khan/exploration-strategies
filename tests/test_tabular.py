import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from exploration.agent import QLearner, schedule
from exploration.environment import GridWorld
from exploration.evaluation import evaluate


ROOT = Path(__file__).resolve().parents[1]


class EnvironmentTests(unittest.TestCase):
    def test_reward_wall_and_terminal_precedence(self):
        env = GridWorld(["S.G", ".#."], horizon=2)
        self.assertEqual(env.shortest_path(), [0, 1, 2])
        s, _ = env.reset(seed=5)
        self.assertEqual(env.transition(s, 0), s)
        self.assertEqual(env.transition(s, 3), s)
        self.assertEqual(env.step(1)[1:4], (0.0, False, False))
        self.assertEqual(env.step(1)[1:4], (1.0, True, False))
        with self.assertRaises(RuntimeError):
            env.step(1)
        self.assertEqual(env.reset(seed=5)[0], s)

    def test_truncation_and_valid_spaces(self):
        env = GridWorld(["S.G", ".#."], horizon=1)
        self.assertEqual(env.step(0)[1:4], (0.0, False, True))
        for s in range(env.n_states):
            for a in range(4):
                self.assertIn(env.transition(s,a), range(env.n_states))
        with self.assertRaises(ValueError):
            env.transition(0, 4)
        with self.assertRaises(ValueError):
            GridWorld(["S#G"])


class LearningTests(unittest.TestCase):
    def test_q_targets_terminal_and_external_time_limit(self):
        agent = QLearner(2, alpha=1, gamma=.9)
        agent.q[1] = [2.0]*4
        agent.update(0, 0, 1, 1, terminated=True)
        agent.update(0, 1, 0, 1, terminated=False)
        self.assertEqual(agent.q[0][0], 1)
        self.assertAlmostEqual(agent.q[0][1], 1.8)

    def test_schedules_and_action_distributions(self):
        cfg = {"kind": "epsilon", "start": 1, "end": .05, "decay_fraction": .5}
        self.assertEqual(schedule(cfg,0,100), 1)
        self.assertAlmostEqual(schedule(cfg,50,100), .05)
        self.assertAlmostEqual(schedule(cfg,100,100), .05)
        agent = QLearner(1)
        agent.q[0] = [1000, 999, 999, 998]
        for kind, val in [("epsilon", .2), ("softmax", .01), ("softmax", 100)]:
            probs = agent.probabilities(0,kind,val)
            self.assertAlmostEqual(sum(probs), 1)
            self.assertTrue(all(0<=p<=1 for p in probs))
            self.assertGreater(probs[0],probs[-1])
        self.assertEqual(agent.probabilities(0,"epsilon",1), [.25]*4)

    def test_evaluation_does_not_change_policy_or_training_rng(self):
        cfg = json.loads((ROOT / "configs/tabular_pilot.json").read_text())
        agent = QLearner(GridWorld(**cfg["environment"]).n_states)
        before = copy.deepcopy(agent.q), agent.rng.getstate()
        a = evaluate(agent,cfg,cfg["conditions"][0],.2)
        b = evaluate(agent,cfg,cfg["conditions"][0],.2)
        self.assertEqual(a,b)
        self.assertEqual(before, (agent.q,agent.rng.getstate()))

    def test_tiny_task_can_learn(self):
        env = GridWorld(["SG"],horizon=8)
        agent = QLearner(env.n_states,seed=2)
        for _ in range(50):
            state,_=env.reset()
            while True:
                action=agent.act(state,"epsilon",.3)
                nxt,reward,term,trunc,_=env.step(action)
                agent.update(state,action,reward,nxt,term)
                state=nxt
                if term or trunc:
                    break
        self.assertEqual(agent.greedy(env.start), 1)

    def test_end_to_end_checkpoint_and_repeatability(self):
        cfg = json.loads((ROOT / "configs/tabular_pilot.json").read_text())
        cfg["experiment"].update(steps=120, eval_every=60, seeds=[4])
        cfg["conditions"] = cfg["conditions"][:1]
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            config_path=tmp/"config.json"
            config_path.write_text(json.dumps(cfg))
            def run(*args):
                return subprocess.check_output([sys.executable,str(ROOT/"scripts/run.py"),*args],text=True,cwd=ROOT,stderr=subprocess.PIPE)
            for folder in ("a","b"):
                run("train","--config",str(config_path),"--output",str(tmp/folder))
            a=tmp/"a/fixed_epsilon/seed_4"
            b=tmp/"b/fixed_epsilon/seed_4"
            checkpoint=json.loads((a/"checkpoint_0000120.json").read_text())
            self.assertEqual(checkpoint,json.loads((b/"checkpoint_0000120.json").read_text()))
            self.assertEqual((a/"evaluations.csv").read_bytes(),(b/"evaluations.csv").read_bytes())
            output=json.loads(run("evaluate","--checkpoint",str(a/"checkpoint_0000120.json")))
            with (a/"evaluations.csv").open() as f:
                last=list(csv.DictReader(f))[-1]
            self.assertEqual(output["greedy_return"],float(last["greedy_return"]))
            self.assertEqual(output["exploring_success"],float(last["exploring_success"]))
            with self.assertRaises(subprocess.CalledProcessError):
                run("train","--config",str(config_path),"--output",str(tmp/"a"))


if __name__ == "__main__":
    unittest.main()
