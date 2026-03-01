from __future__ import annotations

from pathlib import Path

from olympiad_llm.aimo3.v2.meta_learning import get_global_bandit, reset_global_state
from olympiad_llm.aimo3.v2.wickelgren import augment_developer_prompt_with_meta


def test_global_bandit_reconfigures_with_new_hyperparameters(tmp_path: Path) -> None:
    reset_global_state()

    first = get_global_bandit(
        strategy_names=["s1", "s2"],
        exploration_factor=0.25,
        similarity_threshold=0.8,
        experience_file=tmp_path / "a.pkl",
    )

    second = get_global_bandit(
        strategy_names=["s1", "s2"],
        exploration_factor=1.75,
        similarity_threshold=0.55,
        experience_file=tmp_path / "b.pkl",
    )

    assert second is not first
    assert second.exploration_factor == 1.75
    assert second.similarity_threshold == 0.55
    assert second.experience_file == tmp_path / "b.pkl"


def test_preferred_strategy_applies_when_meta_learning_disabled() -> None:
    _prompt, meta = augment_developer_prompt_with_meta(
        "system prompt",
        attempt_index=0,
        problem_text="Find the remainder when n is divided by 7.",
        used_strategies=[],
        meta_learning_enabled=False,
        preferred_strategy="modular_arithmetic",
    )

    assert meta["card"] == "modular_arithmetic"
    assert meta["strategy_selection_method"] == "preferred_strategy"
