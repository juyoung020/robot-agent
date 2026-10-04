//! progress.jsonl 키 이름 — 표준 RL 기록 이름(Stable-Baselines3 꼴 `<묶음>/<이름>`, 2026-10-04 바꿈).
//! 옛 실행 폴더(이 날 이전 키)는 뷰어가 읽을 때 `canon` 으로 새 이름으로 바꾼다(되돌림 호환). 표와 출처는 training/viewer/README.md "용어".
//!
//! 출처: SB3 logger(rollout/ep_rew_mean, rollout/ep_len_mean, rollout/success_rate, time/fps, time/iterations, time/total_timesteps,
//! time/time_elapsed, train/approx_kl, train/clip_fraction, train/entropy_loss, train/explained_variance, train/learning_rate,
//! train/policy_gradient_loss, train/value_loss, train/std, train/loss, eval/success_rate, eval/mean_ep_length, eval/mean_reward).
//! SB3 에 없는 것은 같은 꼴로 지었다(rollout/collision_rate, curriculum/*, dagger/*, train/grad_norm 등).

/// 옛 이름 → (새 이름, 배율). 배율 −1 은 부호를 바꿈(옛 train/entropy = 엔트로피, 새 train/entropy_loss = −엔트로피, SB3 정의).
pub fn canon(k: &str) -> (String, f64) {
    let exact: &[(&str, &str)] = &[
        ("time/iter", "time/iterations"),
        ("time/env_steps", "time/total_timesteps"),
        ("time/wall", "time/time_elapsed"),
        ("time/fps_env", "time/fps"),
        ("time/iters_in_row", "time/iterations_merged"),
        ("rollout/n_eps", "rollout/n_episodes"),
        ("rollout/ep_ret_mean", "rollout/ep_rew_mean"),
        ("rollout/step_rew_mean", "rollout/step_reward_mean"),
        ("rollout_teacher/n_eps", "rollout_teacher/n_episodes"),
        ("train/clip_frac", "train/clip_fraction"),
        ("train/policy_loss", "train/policy_gradient_loss"),
        ("train/lr", "train/learning_rate"),
        ("train/log_std_mean", "train/log_std"),
        ("train/adv_std", "train/advantage_std"),
        ("train/adv_mean", "train/advantage_mean"),
        ("bc/loss", "train/loss"),
        ("bc/flow_loss", "train/flow_loss"),
        ("bc/end_bce", "train/end_bce"),
        ("dagger/disagree", "dagger/action_mse"),
        ("dagger/agree", "dagger/action_mse"),
        ("data/samples", "dagger/dataset_size"),
        ("dagger/new_samples", "dagger/new_samples"),
        ("curr/stage_idx", "curriculum/stage"),
        ("curr/env_stage", "curriculum/env_level"),
        ("curr/goal_known", "curriculum/goal_known_rate"),
        ("curr/completion_start_mean", "curriculum/start_completion_mean"),
    ];
    if k == "train/entropy" {
        return ("train/entropy_loss".into(), -1.0);
    }
    for (a, b) in exact {
        if k == *a {
            return (b.to_string(), 1.0);
        }
    }
    let pre: &[(&str, &str)] = &[
        ("rollout/success/", "rollout/success_rate/"),
        ("rollout_teacher/success/", "rollout_teacher/success_rate/"),
        ("rollout/success_start/", "rollout/success_rate_by_start_map/"),
        ("rollout_teacher/success_start/", "rollout_teacher/success_rate_by_start_map/"),
        ("rollout/ep_len/", "rollout/ep_len_mean/"),
        ("rollout/n_eps/", "rollout/n_episodes/"),
        ("curr/success/", "curriculum/success_rate/"),
        ("curr/collision/", "curriculum/collision_rate/"),
        ("curr/n_eps/", "curriculum/n_episodes/"),
        ("curr/stage_frac/", "curriculum/start_map_fraction/"),
        ("eval/student/success/", "eval/success_rate/"),
        ("eval/student/success_completion/", "eval/success_rate_by_completion/"),
        ("eval/teacher/success/", "eval_teacher/success_rate/"),
        ("eval/teacher/success_completion/", "eval_teacher/success_rate_by_completion/"),
    ];
    for (a, b) in pre {
        if let Some(r) = k.strip_prefix(a) {
            return (format!("{}{}", b, r), 1.0);
        }
    }
    for (who, ns) in [("eval/student/", "eval/"), ("eval/teacher/", "eval_teacher/")] {
        if let Some(r) = k.strip_prefix(who) {
            let r = match r {
                "ep_len_mean" => "mean_ep_length",
                "n_eps" => "n_episodes",
                x => x,
            };
            return (format!("{}{}", ns, r), 1.0);
        }
    }
    (k.to_string(), 1.0)
}

#[cfg(test)]
mod tests {
    use super::canon;
    #[test]
    fn aliases() {
        assert_eq!(canon("rollout/ep_ret_mean").0, "rollout/ep_rew_mean");
        assert_eq!(canon("train/entropy"), ("train/entropy_loss".into(), -1.0));
        assert_eq!(canon("rollout/success/pick").0, "rollout/success_rate/pick");
        assert_eq!(canon("eval/student/ep_len_mean").0, "eval/mean_ep_length");
        assert_eq!(canon("eval/teacher/success/approach").0, "eval_teacher/success_rate/approach");
        assert_eq!(canon("curr/stage_frac/C2").0, "curriculum/start_map_fraction/C2");
        assert_eq!(canon("train/approx_kl").0, "train/approx_kl");
    }
}
