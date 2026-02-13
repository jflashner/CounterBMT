"""End-to-end pipeline orchestration for CounterBMT v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from counter_bmt_v2.causal import DAGBuilder, PromptBNDAGBuilder, SimpleDAGBuilder, SimpleInterventionSampler
from counter_bmt_v2.conditioning import ConditioningModel, DenseConditioningModel
from counter_bmt_v2.config import PipelineConfig
from counter_bmt_v2.contracts import PipelineResult, ScenarioInput
from counter_bmt_v2.judge import MockTrajectoryJudge, TrajectoryJudge
from counter_bmt_v2.perception import GPT4oPerceptionModel, MockPerceptionModel, PerceptionModel
from counter_bmt_v2.rl import compose_reward
from counter_bmt_v2.trajectory_jax import JaxTrajectoryGenerator, TrajectoryGenerator


@dataclass
class CounterBMTPipeline:
    config: PipelineConfig
    perception: PerceptionModel
    dag_builder: DAGBuilder
    sampler: SimpleInterventionSampler
    conditioning: ConditioningModel
    trajectory: TrajectoryGenerator
    judge: TrajectoryJudge

    @classmethod
    def default(cls, config: Optional[PipelineConfig] = None) -> "CounterBMTPipeline":
        cfg = config or PipelineConfig()
        return cls(
            config=cfg,
            perception=MockPerceptionModel(),
            dag_builder=SimpleDAGBuilder(),
            sampler=SimpleInterventionSampler(),
            conditioning=DenseConditioningModel(signal_dim=cfg.conditioning.signal_dim),
            trajectory=JaxTrajectoryGenerator(config=cfg.trajectory),
            judge=MockTrajectoryJudge(),
        )

    @classmethod
    def from_backends(
        cls,
        *,
        config: Optional[PipelineConfig] = None,
        perception_backend: str = "mock",
        dag_backend: str = "simple",
        llm_model: str = "gpt-4o",
        api_key: Optional[str] = None,
        dag_retries: int = 4,
    ) -> "CounterBMTPipeline":
        cfg = config or PipelineConfig()

        if perception_backend == "gpt4o":
            perception: PerceptionModel = GPT4oPerceptionModel(model=llm_model, api_key=api_key)
        else:
            perception = MockPerceptionModel()

        if dag_backend == "promptbn":
            dag_builder: DAGBuilder = PromptBNDAGBuilder(
                model=llm_model,
                api_key=api_key,
                max_retries=dag_retries,
            )
        else:
            dag_builder = SimpleDAGBuilder()

        return cls(
            config=cfg,
            perception=perception,
            dag_builder=dag_builder,
            sampler=SimpleInterventionSampler(),
            conditioning=DenseConditioningModel(signal_dim=cfg.conditioning.signal_dim),
            trajectory=JaxTrajectoryGenerator(config=cfg.trajectory),
            judge=MockTrajectoryJudge(),
        )

    def run(
        self,
        scene: ScenarioInput,
        *,
        n_samples: int = 2,
        seed: int = 0,
        rare: bool = False,
    ) -> PipelineResult:
        features = self.perception.extract(scene)
        dag = self.dag_builder.build(scene, features)
        intervention = self.sampler.sample(dag, rare=rare, seed=seed)
        signal = self.conditioning.build(intervention, dag)
        rollouts = self.trajectory.generate(scene, signal, n_samples=n_samples, seed=seed)

        judge_results = [self.judge.evaluate(intervention, r) for r in rollouts]
        rewards = [compose_reward(j, r, self.config.reward) for j, r in zip(judge_results, rollouts)]

        return PipelineResult(
            scenario_id=scene.scenario_id,
            features=features,
            dag=dag,
            intervention=intervention,
            rollouts=rollouts,
            judge_results=judge_results,
            rewards=rewards,
        )
