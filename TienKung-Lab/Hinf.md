# H-Infinity Disturber Plugin — 技术报告

> 编写日期：2026-03-19
> 作者：ckx
> 基础代码库：TienKung-Lab（IsaacLab + RSL-RL + DreamWaQ）
> 参考论文：*Learning H-Infinity Locomotion Control*

---

## 1. 背景与动机

### 1.1 原始任务：g1_dwaq

`g1_dwaq` 任务使用 **DreamWaQ（DWAQ）** 算法训练 Unitree G1 双足机器人的盲视行走策略。

DreamWaQ 的核心思想：
- 用 β-VAE 将历史本体感知（observation history）编码为连续潜空间特征（latent code）
- Actor 使用 `obs + latent_code` 输出动作，无需高度扫描（blind walking）
- Critic 使用特权信息（privileged obs）估计价值
- 训练算法：DWAQPPO（PPO + VAE autoencoder loss）

### 1.2 问题：鲁棒性不足

标准 PPO 训练的策略在遇到未见过的外力扰动（推力、碰撞）时容易失稳。H-Infinity 控制理论提供了一种系统性的鲁棒化方法：**通过对抗训练，让策略在最坏情况下仍能保持性能**。

### 1.3 解决方案：可学习 Disturber

引入一个与 Actor 对抗的 **Disturber 网络**：
- Disturber 观测当前物理状态，输出对抗外力施加到机器人躯干
- Actor 在受到对抗力的情况下仍需完成速度追踪任务
- 两者通过 **双梯度下降（Dual Gradient Descent）** 交替更新

---

## 2. 理论基础

### 2.1 H-Infinity 问题形式化

$$\min_\pi \max_{d \in \mathcal{D}_\eta} J(\pi, d)$$

其中：
- $\pi$：机器人控制策略（Actor）
- $d$：Disturber 策略，输出外力 $f_t \in \mathbb{R}^3$
- $\mathcal{D}_\eta$：约束集合，$\mathbb{E}[\|f_t\|_2] \leq \eta$（η 为外力上界，单位 N）
- $J(\pi, d)$：累积 tracking error（仅含速度误差，不含辅助奖励）

### 2.2 Lagrangian 松弛（公式 11）

将约束优化转化为无约束问题：

$$\mathcal{L}(\pi, d, \lambda) = J(\pi, d) + \lambda \left(\mathbb{E}[\|f_t\|_2] - \eta\right)$$

- $\lambda \geq 0$：Lagrangian 乘子，自动调节约束松紧度
- $\lambda$ 增大 → 约束更严格（力更小）
- $\lambda$ 减小 → 约束更宽松（力更大）

### 2.3 双梯度下降更新顺序（公式 12）

每次 iteration 严格按以下顺序更新：

1. **更新 π（Actor）**：最小化 PPO loss + λ·Lagrangian（DWAQPPO.update）
2. **更新 d（Disturber）**：最大化 cost advantages（PPO clipping）
3. **更新 λ**：梯度上升，若 $\|f\| > \eta$ 则增大 λ

### 2.4 Cost 函数定义

$$C_t = w_{lin} \cdot \|v_{cmd,xy} - v_{actual,xy}\|^2 + w_{ang} \cdot \|\omega_{cmd,z} - \omega_{actual,z}\|^2$$

**严格只包含 tracking error**，不含扭矩惩罚、平滑度等辅助项。这是论文的关键要求：Disturber 只针对"让机器人偏离速度命令"这一目标进行优化。

---

## 3. 代码架构

### 3.1 新增/修改文件总览

```
TienKung-Lab/
├── rsl_rl/rsl_rl/
│   ├── algorithms/
│   │   ├── hinf_disturber.py          ← [新增] H-Infinity 核心实现
│   │   └── __init__.py                ← [修改] 导出新类
│   └── runners/
│       ├── ppo_trainer_patch.py       ← [新增] DWAQHInfRunner
│       └── __init__.py                ← [修改] 导出新 Runner
├── legged_lab/
│   ├── envs/
│   │   ├── g1/
│   │   │   └── g1_dwaq_config.py      ← [修改] 添加 disturber 配置 + G1DwaqHInfAgentCfg
│   │   └── __init__.py                ← [修改] 注册 g1_dwaq_hinf 任务
│   └── scripts/
│       └── train.py                   ← [修改] 导入 DWAQHInfRunner 供 eval() 使用
```

### 3.2 文件详细说明

---

#### `rsl_rl/rsl_rl/algorithms/hinf_disturber.py` — 核心实现

包含 5 个类：

**`HInfDisturberCfg`（dataclass）**

超参数配置，所有字段均有默认值，通过 `G1DwaqAgentCfg.disturber` 字典传入：

| 字段 | 默认值 | 含义 |
|------|--------|------|
| `disturber_obs_dim` | 82 | Disturber 输入维度（与 actor_obs 一致，gait_phase=True 时为 82） |
| `eta` | 50.0 | 外力 L2 范数上界（N） |
| `apply_body_name` | `"torso_link"` | 施力 body 名称 |
| `warmup_iterations` | 500 | 前 N 次 iteration 不激活 Disturber |
| `actor_hidden_dims` | [256,128,64] | Disturber Actor 网络结构 |
| `critic_hidden_dims` | [256,128,64] | Disturber Critic 网络结构 |
| `force_dim` | 3 | 外力维度（xyz） |
| `disturber_lr` | 3e-4 | Disturber 网络学习率 |
| `lagrangian_lr` | 1e-3 | λ 学习率 |
| `lambda_init` | 1.0 | λ 初始值 |
| `lambda_min/max` | 0.0 / 10.0 | λ 截断范围 |
| `clip_param` | 0.2 | PPO clipping 系数 |
| `gamma` | 0.99 | 折扣因子 |
| `lam` | 0.95 | GAE lambda |
| `cost_lin_vel_weight` | 1.0 | C_t 线速度误差权重 |
| `cost_ang_vel_weight` | 0.5 | C_t 角速度误差权重 |

**`DisturberActor`（nn.Module）**

- 输入：`disturber_obs`（纯物理状态，已 detach）
- 输出：对抗力 $f_t \in \mathbb{R}^3$，经过 L2 范数截断：
  ```python
  scale = eta / max(norm, eta)  # 确保 ||f|| ≤ η
  f = raw_output * scale
  ```
- 网络结构：MLP，默认 [256, 128, 64]，ELU 激活

**`DisturberCritic`（nn.Module）**

- 输入：`disturber_obs`
- 输出：cost value 估计 $V(s)$，用于 GAE
- 网络结构：MLP，默认 [256, 128, 64]，ELU 激活

**`DisturberRolloutStorage`**

- 存储一个 rollout 的所有数据：obs、forces、costs、dones、values、log_probs、mu、sigma
- 形状：`[T, N, ...]`，T = num_steps_per_env，N = num_envs
- `clear()` 重置 `self.step = 0`（每次 update 后必须调用）
- `compute_returns()` 使用 GAE 计算 cost advantages 和 returns
- `mini_batch_generator()` 随机打乱并生成 mini-batch

**`HInfDisturberPlugin`（主类）**

对外接口：

| 方法 | 调用时机 | 说明 |
|------|----------|------|
| `step(disturber_obs)` | rollout 每步，env.step() 之前 | 生成外力并注入 IsaacLab |
| `record(obs, costs, dones)` | rollout 每步，env.step() 之后 | 记录数据到 storage |
| `compute_returns(last_obs)` | rollout 结束后 | 计算 GAE |
| `update()` | compute_returns 之后 | 双梯度下降更新 |
| `is_active()` | 内部 | `iteration >= warmup_iterations` |
| `train() / eval()` | Runner 调用 | 切换训练/推理模式 |
| `state_dict() / load_state_dict()` | checkpoint | 保存/加载状态 |

**外力注入实现：**

```python
# 纯 GPU Tensor 操作，无 for 循环
forces_world = torch.zeros(num_envs, num_bodies, 3, device=device)
body_idx = robot.find_bodies(apply_body_name)[0]
forces_world[:, body_idx, :] = f_t  # [N, 3]
robot.set_external_force_and_torque(forces_world, torques_world)
```

---

#### `rsl_rl/rsl_rl/runners/ppo_trainer_patch.py` — DWAQHInfRunner

继承 `DWAQOnPolicyRunner`，覆写 `learn()` 循环。

**Rollout 阶段新增步骤（每个 physics step）：**

```
Step 1: _extract_disturber_obs(obs)     → 提取纯物理状态（detach）
Step 2: disturber.step(disturber_obs)   → 生成并注入对抗力
Step 3: alg.act() + env.step()          → 原有 PPO 逻辑（外力已生效）
Step 4: _compute_cost_from_env()        → 计算 C_t（仅 tracking error）
Step 5: disturber.record(obs, cost, done)
```

**Update 阶段新增步骤（每个 iteration）：**

```
Step 6: alg.compute_returns() + alg.update()    → Actor PPO 更新（原有）
Step 7: disturber.compute_returns(last_obs)      → Disturber GAE
Step 8: disturber.update()                       → Disturber + λ 更新
```

**关键辅助方法：**

- `_extract_disturber_obs(obs, critic_obs)`：从 actor_obs 提取 Disturber 输入，若维度不匹配则截断或零填充
- `_compute_cost_from_env(obs, critic_obs)`：直接从 `env.command_generator` 和 `env.robot.data` 获取速度，计算 C_t
- `_log_hinf(locs, loss_dict, disturber_loss_dict)`：将 `loss_dict` 的值注入 `locs`（父类 `log()` 需要），并额外写入 `HInf/` TensorBoard 指标
- `save/load`：在父类 checkpoint 基础上额外保存/加载 `*_disturber.pt`
- `train_mode/eval_mode`：加 `hasattr(self, 'disturber')` 保护，防止父类 `__init__` 调用时 disturber 尚未初始化

---

#### `legged_lab/envs/g1/g1_dwaq_config.py` — 配置修改

**`G1DwaqAgentCfg.__post_init__` 新增 `self.disturber` 字典：**

```python
self.disturber = {
    "disturber_obs_dim": 82,         # gait_phase=True → 82 维
    "eta": 50.0,                     # 外力上界（N）
    "apply_body_name": "torso_link",
    "warmup_iterations": 500,
    ...
}
```

`runner_class_name` 改为 `"DWAQHInfRunner"`（原为 `"DWAQOnPolicyRunner"`）。

**新增 `G1DwaqHInfAgentCfg`：**

```python
@configclass
class G1DwaqHInfAgentCfg(G1DwaqAgentCfg):
    experiment_name: str = "g1_dwaq_hinf"
    wandb_project: str = "g1_dwaq_hinf"
```

完全继承父类，只改 `experiment_name`，使日志写入 `logs/g1_dwaq_hinf/`。

---

#### `legged_lab/envs/__init__.py` — 任务注册

```python
task_registry.register("g1_dwaq_hinf", G1DwaqEnv, G1DwaqEnvCfg(), G1DwaqHInfAgentCfg())
```

与 `g1_dwaq` 使用完全相同的环境，只是 Agent 配置不同（日志目录区分）。

---

#### `legged_lab/scripts/train.py` — 导入修复

```python
from rsl_rl.runners import DWAQHInfRunner  # noqa: F401
```

`train.py` 用 `eval(agent_cfg.runner_class_name)` 实例化 Runner，必须在当前命名空间中有 `DWAQHInfRunner`，否则报 `NameError`。

---

## 4. Bug 修复记录

### Bug 1：NameError: DWAQHInfRunner

**现象：** 训练启动时报 `NameError: name 'DWAQHInfRunner' is not defined`

**根因：** `train.py` 第99行 `eval(agent_cfg.runner_class_name)` 在当前命名空间中找不到 `DWAQHInfRunner`

**修复：** 在 `train.py` 顶部添加 `from rsl_rl.runners import DWAQHInfRunner`

---

### Bug 2：DisturberRolloutStorage overflow

**现象：** 第二次 iteration 时报 `RuntimeError: DisturberRolloutStorage overflow`

**根因：** `update()` 在 warmup 期间（`is_active() == False`）提前 return，没有调用 `storage.clear()`，导致 storage 的 `step` 指针不断累积，第二次 rollout 写入时越界

**修复：** 在 warmup 提前返回前加 `self.storage.clear()`：

```python
if not self.is_active():
    self.storage.clear()  # [Fix] warmup 期间也必须清空
    self.iteration += 1
    return {...}
```

---

### Bug 3：train_mode 初始化顺序

**现象：** 父类 `DWAQOnPolicyRunner.__init__` 内部调用 `self.train_mode()`，此时子类的 `self.disturber` 尚未初始化，导致 `AttributeError`

**修复：** 在 `train_mode` 和 `eval_mode` 中加 `hasattr` 保护：

```python
def train_mode(self):
    super().train_mode()
    if hasattr(self, "disturber"):
        self.disturber.train()
```

---

### Bug 4：_log_hinf 缺少必要键

**现象：** 父类 `log()` 需要 `locs` 中包含 `mean_value_loss`、`mean_surrogate_loss`、`mean_autoenc_loss`，但 `learn()` 的 `locals()` 里没有这些键

**修复：** 在 `_log_hinf` 中从 `loss_dict` 补充这些键：

```python
locs["mean_value_loss"] = loss_dict["value_function"]
locs["mean_surrogate_loss"] = loss_dict["surrogate"]
locs["mean_autoenc_loss"] = loss_dict["autoencoder"]
```

---

## 5. 使用方法

### 5.1 从头训练

```bash
python legged_lab/scripts/train.py \
  --task g1_dwaq_hinf \
  --num_envs 4096 \
  --headless \
  --logger tensorboard
```

### 5.2 从 g1_dwaq checkpoint 继续训练

先将 checkpoint 复制到 `logs/g1_dwaq_hinf/` 下（或直接指定路径）：

```bash
# 方法：直接 resume（logs/g1_dwaq_hinf/ 下已有 run 目录）
python legged_lab/scripts/train.py \
  --task g1_dwaq_hinf \
  --num_envs 4096 \
  --headless \
  --resume True \
  --load_run 2026-01-16_00-46-00 \
  --checkpoint model_9999.pt
```

**参数说明：**
- `--resume True`：启用 checkpoint 加载（注意是 `True` 字符串，不是 `RESUME`）
- `--load_run`：`logs/g1_dwaq_hinf/` 下的子目录名（时间戳格式）
- `--checkpoint`：`.pt` 文件名（不是 `--load_checkpoint`）

### 5.3 监控训练

```bash
tensorboard --logdir logs/g1_dwaq_hinf
```

**关键指标：**

| 指标 | 正常范围 | 含义 |
|------|----------|------|
| `HInf/disturber/mean_force_norm` | warmup 后从 0 上升，趋向 η=50 | Disturber 激活并施加外力 |
| `HInf/disturber/lambda` | 0~10，动态调整 | Lagrangian 乘子 |
| `HInf/disturber/actor_loss` | 非零 | Disturber 网络在更新 |
| `HInf/disturber/mean_cost` | 正值 | tracking error（Disturber 试图最大化） |
| `Train/mean_reward` | 激活后略降，随后恢复 | 鲁棒性提升的标志 |

**Disturber 激活时机：** `iteration >= warmup_iterations`（默认 500）。从 checkpoint 9999 继续训练时，激活时间为 iteration 10499。

### 5.4 与 g1_dwaq 的区别

| 项目 | g1_dwaq | g1_dwaq_hinf |
|------|---------|--------------|
| Runner | DWAQHInfRunner | DWAQHInfRunner |
| 环境 | G1DwaqEnv | G1DwaqEnv（完全相同） |
| 奖励 | G1DwaqRewardCfg | G1DwaqRewardCfg（完全相同） |
| 日志目录 | logs/g1_dwaq/ | logs/g1_dwaq_hinf/ |
| Disturber | 无 | 有（warmup 500 iter 后激活） |

---

## 6. 已知限制与后续工作

### 6.1 API 废弃警告

`set_external_force_and_torque` 在未来 IsaacLab 版本将被废弃，需迁移到：

```python
robot.permanent_wrench_composer.set_forces_and_torques(forces, torques, body_ids, is_global=True)
```

### 6.2 超参数调优建议

- `eta`：从 50N 开始，逐步增大到 100N、200N，观察机器人是否能适应
- `warmup_iterations`：500 是保守值，可以减小到 200 加速对抗训练启动
- `disturber_lr`：若 Disturber 收敛过慢，可适当增大到 1e-3

### 6.3 Disturber obs_dim 与 gait_phase 的关联

`disturber_obs_dim` 必须与 `G1DwaqEnvCfg.robot.gait_phase.enable` 保持一致：
- `gait_phase.enable = True` → `disturber_obs_dim = 82`（78基础 + sin/cos各2）
- `gait_phase.enable = False` → `disturber_obs_dim = 78`

修改 gait_phase 配置时，必须同步修改 `G1DwaqAgentCfg.disturber["disturber_obs_dim"]`。

---

## 7. 文件修改 diff 摘要

### `rsl_rl/rsl_rl/algorithms/__init__.py`

```python
# [H-Infinity Plugin] 新增
from .hinf_disturber import HInfDisturberPlugin, HInfDisturberCfg
```

### `rsl_rl/rsl_rl/runners/__init__.py`

```python
# [H-Infinity Plugin] 新增
from .ppo_trainer_patch import DWAQHInfRunner
```

### `legged_lab/scripts/train.py`

```python
# [H-Infinity Plugin] 新增，供 eval(runner_class_name) 使用
from rsl_rl.runners import DWAQHInfRunner  # noqa: F401
```

### `legged_lab/envs/g1/g1_dwaq_config.py`

```python
# G1DwaqAgentCfg 中：
runner_class_name: str = "DWAQHInfRunner"  # 原为 "DWAQOnPolicyRunner"

# __post_init__ 中新增：
self.disturber = { "eta": 50.0, "warmup_iterations": 500, ... }

# 新增类：
@configclass
class G1DwaqHInfAgentCfg(G1DwaqAgentCfg):
    experiment_name: str = "g1_dwaq_hinf"
```

### `legged_lab/envs/__init__.py`

```python
# 新增任务注册：
task_registry.register("g1_dwaq_hinf", G1DwaqEnv, G1DwaqEnvCfg(), G1DwaqHInfAgentCfg())
```
